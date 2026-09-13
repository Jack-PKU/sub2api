package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tiktoken-go/tokenizer"
	"go.uber.org/zap"
)

const (
	openAIResponsesInputItemTokenOverhead = 3
	openAIResponsesContentPartOverhead    = 1
	openAIInputTokensFallbackMinimum      = 1

	// openAIInputTokensUnsupportedTTL bounds how long a per-account "platform
	// input_tokens is unusable" verdict suppresses the upstream probe. Long
	// enough that a Claude Code session (and its end-of-turn burst of 80+
	// count_tokens calls) costs one probe at most, short enough that a
	// re-authorized credential recovers on its own.
	openAIInputTokensUnsupportedTTL = 6 * time.Hour

	// openAIInputTokensMemoizedStatus marks a fallback that skipped the upstream
	// probe entirely. It is not an HTTP status: 0 distinguishes these log lines
	// from a fallback triggered by a real upstream 401/403/404.
	openAIInputTokensMemoizedStatus = 0
)

type openAIInputTokensCountRequest struct {
	Model        string                    `json:"model"`
	Instructions string                    `json:"instructions,omitempty"`
	Input        json.RawMessage           `json:"input,omitempty"`
	Tools        []apicompat.ResponsesTool `json:"tools,omitempty"`
	ToolChoice   json.RawMessage           `json:"tool_choice,omitempty"`
}

type openAIInputTokensCountPrepared struct {
	Request         openAIInputTokensCountRequest
	OriginalModel   string
	NormalizedModel string
	BillingModel    string
	UpstreamModel   string
}

// EstimateGrokCountTokens estimates an Anthropic-compatible count_tokens request
// locally. Grok does not expose a compatible token-counting endpoint, so this
// path deliberately avoids account selection, credentials, and upstream calls.
func EstimateGrokCountTokens(body []byte) (int, error) {
	var anthropicReq apicompat.AnthropicRequest
	if err := json.Unmarshal(body, &anthropicReq); err != nil {
		return 0, fmt.Errorf("parse anthropic count_tokens request: %w", err)
	}
	if strings.TrimSpace(anthropicReq.Model) == "" {
		return 0, fmt.Errorf("parse anthropic count_tokens request: model is required")
	}

	responsesReq, err := apicompat.AnthropicToResponses(&anthropicReq)
	if err != nil {
		return 0, fmt.Errorf("convert anthropic request to responses: %w", err)
	}

	estimated, err := estimateOpenAIInputTokens(openAIInputTokensCountRequest{
		Model:        anthropicReq.Model,
		Instructions: responsesReq.Instructions,
		Input:        responsesReq.Input,
		Tools:        responsesReq.Tools,
		ToolChoice:   responsesReq.ToolChoice,
	})
	if err != nil {
		return 0, fmt.Errorf("estimate grok input tokens: %w", err)
	}
	if estimated < openAIInputTokensFallbackMinimum {
		estimated = openAIInputTokensFallbackMinimum
	}
	return estimated, nil
}

// ForwardCountTokensAsAnthropic bridges Anthropic /v1/messages/count_tokens to
// OpenAI POST /v1/responses/input_tokens and returns Anthropic-compatible output.
func (s *OpenAIGatewayService) ForwardCountTokensAsAnthropic(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
	defaultMappedModel string,
) error {
	if account == nil {
		writeAnthropicCountTokensError(c, http.StatusServiceUnavailable, "api_error", "No available OpenAI accounts")
		return fmt.Errorf("count_tokens: missing account")
	}

	prepared, err := prepareOpenAIInputTokensCountRequest(body, account, defaultMappedModel)
	if err != nil {
		writeAnthropicCountTokensError(c, http.StatusBadRequest, "invalid_request_error", "Failed to parse request body")
		return err
	}

	upstreamBody, err := marshalOpenAIUpstreamJSON(prepared.Request)
	if err != nil {
		writeAnthropicCountTokensError(c, http.StatusInternalServerError, "api_error", "Failed to build request")
		return fmt.Errorf("marshal openai input_tokens body: %w", err)
	}

	logger.L().Debug("openai count_tokens: model mapping applied",
		zap.Int64("account_id", account.ID),
		zap.String("original_model", prepared.OriginalModel),
		zap.String("normalized_model", prepared.NormalizedModel),
		zap.String("billing_model", prepared.BillingModel),
		zap.String("upstream_model", prepared.UpstreamModel),
	)

	// A ChatGPT subscription OAuth credential carries no api.responses.write
	// scope, so the platform input_tokens endpoint answers 401 for every single
	// call. Claude Code issues count_tokens in bursts (80+ within a second at
	// the end of a turn), so re-probing per call spent one doomed upstream
	// round trip each time. Once an account has answered "unsupported", serve
	// the local estimate directly until the verdict expires; the probe resumes
	// afterwards so a credential that later gains the scope is picked up
	// instead of being permanently downgraded.
	serveLocally, releaseProbe := s.claimOpenAIInputTokensProbe(account)
	if serveLocally {
		writeOpenAIOAuthInputTokensFallback(c, account, prepared, openAIInputTokensMemoizedStatus)
		return nil
	}
	defer releaseProbe()

	token, _, err := s.GetAccessToken(ctx, account)
	if err != nil {
		writeAnthropicCountTokensError(c, http.StatusBadGateway, "upstream_error", "Failed to get access token")
		return fmt.Errorf("get access token: %w", err)
	}

	upstreamReq, err := s.buildInputTokensUpstreamRequest(ctx, c, account, upstreamBody, token)
	if err != nil {
		writeAnthropicCountTokensError(c, http.StatusInternalServerError, "api_error", "Failed to build request")
		return fmt.Errorf("build input_tokens request: %w", err)
	}

	proxyURL := ""
	if account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}
	resp, err := s.httpUpstream.Do(upstreamReq, proxyURL, account.ID, account.Concurrency)
	if err != nil {
		safeErr := sanitizeUpstreamErrorMessage(err.Error())
		setOpsUpstreamError(c, 0, safeErr, "")
		writeAnthropicCountTokensError(c, http.StatusBadGateway, "upstream_error", "Upstream request failed")
		return fmt.Errorf("openai input_tokens upstream request failed: %s", safeErr)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		writeAnthropicCountTokensError(c, http.StatusBadGateway, "upstream_error", "Failed to read response")
		return fmt.Errorf("read input_tokens response: %w", err)
	}

	if resp.StatusCode >= 400 {
		upstreamMsg := sanitizeUpstreamErrorMessage(strings.TrimSpace(extractUpstreamErrorMessage(respBody)))
		if account.Type == AccountTypeOAuth && isOpenAIOAuthInputTokensUnsupported(resp.StatusCode, respBody) {
			s.rememberOpenAIInputTokensUnsupported(account, resp.StatusCode)
			writeOpenAIOAuthInputTokensFallback(c, account, prepared, resp.StatusCode)
			return nil
		}

		if s.rateLimitService != nil {
			s.rateLimitService.HandleUpstreamError(ctx, account, resp.StatusCode, resp.Header, respBody)
		}

		if isOpenAIInputTokensUnsupported(resp.StatusCode, respBody) {
			writeAnthropicCountTokensError(c, http.StatusNotFound, "not_found_error", "Token counting is not supported by upstream")
			return nil
		}

		upstreamDetail := ""
		if s.cfg != nil && s.cfg.Gateway.LogUpstreamErrorBody {
			maxBytes := s.cfg.Gateway.LogUpstreamErrorBodyMaxBytes
			if maxBytes <= 0 {
				maxBytes = 2048
			}
			upstreamDetail = truncateString(string(respBody), maxBytes)
		}
		setOpsUpstreamError(c, resp.StatusCode, upstreamMsg, upstreamDetail)

		errMsg := "Upstream request failed"
		switch resp.StatusCode {
		case 429:
			errMsg = "Rate limit exceeded"
		case 500, 502, 503, 504, 529:
			errMsg = "Upstream service temporarily unavailable"
		}
		writeAnthropicCountTokensError(c, resp.StatusCode, "upstream_error", errMsg)
		if upstreamMsg == "" {
			return fmt.Errorf("input_tokens upstream error: %d", resp.StatusCode)
		}
		return fmt.Errorf("input_tokens upstream error: %d message=%s", resp.StatusCode, upstreamMsg)
	}

	inputTokens := gjson.GetBytes(respBody, "input_tokens")
	if !inputTokens.Exists() {
		writeAnthropicCountTokensError(c, http.StatusBadGateway, "upstream_error", "Upstream response missing input_tokens")
		return fmt.Errorf("input_tokens response missing input_tokens field")
	}
	// The endpoint answered for this credential: stop collapsing later bursts so
	// every caller keeps getting the exact upstream count.
	s.rememberOpenAIInputTokensSupported(account)

	c.JSON(http.StatusOK, gin.H{
		"input_tokens": int(inputTokens.Int()),
	})
	return nil
}

func prepareOpenAIInputTokensCountRequest(
	body []byte,
	account *Account,
	defaultMappedModel string,
) (*openAIInputTokensCountPrepared, error) {
	var anthropicReq apicompat.AnthropicRequest
	if err := json.Unmarshal(body, &anthropicReq); err != nil {
		return nil, fmt.Errorf("parse anthropic count_tokens request: %w", err)
	}

	originalModel := anthropicReq.Model
	applyOpenAICompatModelNormalization(&anthropicReq)
	normalizedModel := anthropicReq.Model
	billingModel := resolveOpenAIForwardModel(account, normalizedModel, strings.TrimSpace(defaultMappedModel))
	upstreamModel := normalizeOpenAIModelForUpstream(account, billingModel)

	responsesReq, err := apicompat.AnthropicToResponses(&anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("convert anthropic request to responses: %w", err)
	}

	return &openAIInputTokensCountPrepared{
		Request: openAIInputTokensCountRequest{
			Model:        upstreamModel,
			Instructions: responsesReq.Instructions,
			Input:        responsesReq.Input,
			Tools:        responsesReq.Tools,
			ToolChoice:   responsesReq.ToolChoice,
		},
		OriginalModel:   originalModel,
		NormalizedModel: normalizedModel,
		BillingModel:    billingModel,
		UpstreamModel:   upstreamModel,
	}, nil
}

func (s *OpenAIGatewayService) buildInputTokensUpstreamRequest(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
	token string,
) (*http.Request, error) {
	targetURL := openaiPlatformAPIInputTokensURL
	if account.Type == AccountTypeAPIKey {
		if baseURL := account.GetOpenAIBaseURL(); strings.TrimSpace(baseURL) != "" {
			validatedURL, err := s.validateUpstreamBaseURL(baseURL)
			if err != nil {
				return nil, err
			}
			targetURL = buildOpenAIResponsesInputTokensURL(validatedURL)
		}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, targetURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req = req.WithContext(WithHTTPUpstreamProfile(req.Context(), HTTPUpstreamProfileOpenAI))
	authHeaders, err := s.buildOpenAIAuthenticationHeaders(ctx, account, token)
	if err != nil {
		return nil, err
	}
	for key, values := range authHeaders {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}
	req.Header.Set("content-type", "application/json")
	req.Header.Set("accept", "application/json")

	if c != nil && c.Request != nil {
		for key, values := range c.Request.Header {
			lower := strings.ToLower(strings.TrimSpace(key))
			if lower != "user-agent" && lower != "accept-language" {
				continue
			}
			for _, v := range values {
				req.Header.Add(key, v)
			}
		}
	}

	// 账号级请求头覆写（仅 openai api_key 账号启用时生效；OAuth 路径 no-op）
	account.ApplyHeaderOverrides(req.Header)

	return req, nil
}

func writeAnthropicCountTokensError(c *gin.Context, status int, errType, message string) {
	c.JSON(status, gin.H{
		"type": "error",
		"error": gin.H{
			"type":    errType,
			"message": message,
		},
	})
}

// openAIInputTokensProbeMark is a sentinel stored in the per-account probe state.
type openAIInputTokensProbeMark int

const (
	// openAIInputTokensProbeRunning means one caller is probing upstream right now
	// and no verdict exists yet for this account.
	openAIInputTokensProbeRunning openAIInputTokensProbeMark = iota
	// openAIInputTokensProbeSupported means a probe succeeded, so the endpoint is
	// real for this credential and every caller should keep getting exact counts.
	openAIInputTokensProbeSupported
)

// claimOpenAIInputTokensProbe decides whether this request may probe the platform
// input_tokens endpoint, and returns a release for the caller that claimed it.
//
// The account state is one of: absent (never probed), running (a probe is in
// flight and nothing is known yet), a time.Time (probed unsupported, valid until
// then), or supported (probe succeeded).
//
// The running state exists because a coding client issues count_tokens in bursts
// of 80+ parallel calls at the start of a turn. A verdict recorded only on
// completion is still empty when the whole burst reads it, so every call in that
// burst probed a doomed endpoint - 79 of them in one observed two-second window.
// Collapsing the burst to a single probe is why concurrent callers answer from
// the local estimator instead of waiting: count_tokens is an estimate-tolerant
// convenience endpoint with a local fallback by design, so a wait on a round trip
// that is about to fail costs latency and buys nothing. An account whose probe
// actually succeeds is marked supported and never skips again, so a working
// endpoint keeps returning exact counts after the first burst.
//
// Only OAuth accounts take part. API-key accounts reach the endpoint normally and
// must never be downgraded to a local estimate.
func (s *OpenAIGatewayService) claimOpenAIInputTokensProbe(account *Account) (serveLocally bool, release func()) {
	noop := func() {}
	if account == nil || account.Type != AccountTypeOAuth {
		return false, noop
	}
	for {
		actual, loaded := s.openaiInputTokensProbeState.LoadOrStore(account.ID, openAIInputTokensProbeRunning)
		if !loaded {
			// We claimed the probe. Clear the marker only if no verdict replaced
			// it, so a probe that ends without recording one (network error) does
			// not wedge the account into "running" forever.
			return false, func() {
				s.openaiInputTokensProbeState.CompareAndDelete(account.ID, openAIInputTokensProbeRunning)
			}
		}
		switch verdict := actual.(type) {
		case openAIInputTokensProbeMark:
			if verdict == openAIInputTokensProbeSupported {
				return false, noop
			}
			return true, noop
		case time.Time:
			if time.Now().After(verdict) {
				// Expired. Drop it and retry the claim so exactly one caller
				// re-probes; a racing caller either wins the claim or reads the
				// running marker.
				s.openaiInputTokensProbeState.CompareAndDelete(account.ID, verdict)
				continue
			}
			return true, noop
		default:
			s.openaiInputTokensProbeState.Delete(account.ID)
			continue
		}
	}
}

// rememberOpenAIInputTokensUnsupported records that this account's credential
// cannot use the platform input_tokens endpoint. The verdict is deliberately
// time-bounded rather than permanent so a re-authorized credential recovers
// without a gateway restart.
func (s *OpenAIGatewayService) rememberOpenAIInputTokensUnsupported(account *Account, statusCode int) {
	if account == nil || account.Type != AccountTypeOAuth {
		return
	}
	if previous, existed := s.openaiInputTokensProbeState.Load(account.ID); !existed || previous == any(openAIInputTokensProbeRunning) {
		logger.L().Info("openai count_tokens: memoizing input_tokens unsupported for account",
			zap.Int64("account_id", account.ID),
			zap.Int("upstream_status", statusCode),
			zap.Duration("ttl", openAIInputTokensUnsupportedTTL),
		)
	}
	s.openaiInputTokensProbeState.Store(account.ID, time.Now().Add(openAIInputTokensUnsupportedTTL))
}

// rememberOpenAIInputTokensSupported records that the endpoint answered for this
// credential, so later bursts stop skipping the probe and keep exact counts.
func (s *OpenAIGatewayService) rememberOpenAIInputTokensSupported(account *Account) {
	if account == nil || account.Type != AccountTypeOAuth {
		return
	}
	s.openaiInputTokensProbeState.Store(account.ID, openAIInputTokensProbeSupported)
}

func isOpenAIInputTokensUnsupported(statusCode int, body []byte) bool {
	if statusCode != http.StatusNotFound {
		return false
	}
	msg := strings.ToLower(strings.TrimSpace(extractUpstreamErrorMessage(body)))
	return strings.Contains(msg, "input_tokens") && strings.Contains(msg, "not found")
}

func writeOpenAIOAuthInputTokensFallback(c *gin.Context, account *Account, prepared *openAIInputTokensCountPrepared, statusCode int) {
	estimated := openAIInputTokensFallbackMinimum
	if got, err := estimateOpenAIInputTokens(prepared.Request); err == nil {
		if got > 0 {
			estimated = got
		}
		logger.L().Info("openai count_tokens: oauth fallback to local tiktoken estimate",
			zap.Int64("account_id", account.ID),
			zap.Int("upstream_status", statusCode),
			zap.Int("estimated_input_tokens", estimated),
			zap.String("upstream_model", prepared.UpstreamModel),
		)
	} else {
		logger.L().Warn("openai count_tokens: oauth local tiktoken fallback failed, using minimum estimate",
			zap.Int64("account_id", account.ID),
			zap.Int("upstream_status", statusCode),
			zap.Int("estimated_input_tokens", estimated),
			zap.String("upstream_model", prepared.UpstreamModel),
			zap.Error(err),
		)
	}

	c.JSON(http.StatusOK, gin.H{
		"input_tokens": estimated,
	})
}

func isOpenAIOAuthInputTokensUnsupported(statusCode int, body []byte) bool {
	switch statusCode {
	case http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound:
	default:
		return false
	}

	bodyLower := strings.ToLower(string(body))
	msg := strings.ToLower(strings.TrimSpace(extractUpstreamErrorMessage(body)))
	code := strings.ToLower(strings.TrimSpace(extractUpstreamErrorCode(body)))

	if code == "missing_scope" ||
		strings.Contains(bodyLower, "api.responses.write") ||
		strings.Contains(bodyLower, "missing scopes") ||
		strings.Contains(bodyLower, "insufficient_scope") {
		return true
	}

	if statusCode == http.StatusNotFound && isOpenAIInputTokensUnsupported(statusCode, body) {
		return true
	}

	// OAuth's platform endpoint can be blocked by an upstream proxy before it
	// reaches the API and return an HTML 403 page without a structured error.
	// Treat that endpoint-level response like the other unsupported cases so
	// count_tokens remains a local, non-health-affecting convenience request.
	if statusCode == http.StatusForbidden && isHTMLResponse(body) {
		return true
	}

	return strings.Contains(msg, "input_tokens") &&
		(strings.Contains(msg, "not found") ||
			strings.Contains(msg, "not supported") ||
			strings.Contains(msg, "unsupported"))
}

func isHTMLResponse(body []byte) bool {
	trimmed := strings.TrimSpace(strings.ToLower(string(body)))
	return strings.HasPrefix(trimmed, "<!doctype html") ||
		strings.HasPrefix(trimmed, "<html")
}

func estimateOpenAIInputTokens(req openAIInputTokensCountRequest) (int, error) {
	codec, err := openAIInputTokensCodecForModel(req.Model)
	if err != nil {
		return 0, err
	}

	total := 0
	addCount := func(text string) error {
		text = strings.TrimSpace(text)
		if text == "" {
			return nil
		}
		n, err := codec.Count(text)
		if err != nil {
			return err
		}
		total += n
		return nil
	}

	if err := addCount(req.Instructions); err != nil {
		return 0, err
	}
	inputTokens, err := estimateOpenAIInputTokensForInput(codec, req.Input)
	if err != nil {
		return 0, err
	}
	total += inputTokens

	for _, tool := range req.Tools {
		raw, err := marshalOpenAIUpstreamJSON(tool)
		if err != nil {
			return 0, err
		}
		if err := addCount(string(raw)); err != nil {
			return 0, err
		}
	}
	if len(req.ToolChoice) > 0 {
		compacted, err := compactOpenAIInputTokensJSON(req.ToolChoice)
		if err != nil {
			return 0, err
		}
		if err := addCount(compacted); err != nil {
			return 0, err
		}
	}

	if total < 0 {
		return 0, nil
	}
	return total, nil
}

func estimateOpenAIInputTokensForInput(codec tokenizer.Codec, raw json.RawMessage) (int, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return 0, nil
	}

	var plainText string
	if err := json.Unmarshal(raw, &plainText); err == nil {
		return codec.Count(plainText)
	}

	var items []apicompat.ResponsesInputItem
	if err := json.Unmarshal(raw, &items); err == nil {
		return estimateOpenAIInputTokensForInputItems(codec, items)
	}

	compacted, err := compactOpenAIInputTokensJSON(raw)
	if err != nil {
		return 0, err
	}
	return codec.Count(compacted)
}

func estimateOpenAIInputTokensForInputItems(codec tokenizer.Codec, items []apicompat.ResponsesInputItem) (int, error) {
	total := 0
	countText := func(text string) error {
		text = strings.TrimSpace(text)
		if text == "" {
			return nil
		}
		n, err := codec.Count(text)
		if err != nil {
			return err
		}
		total += n
		return nil
	}

	for _, item := range items {
		total += openAIResponsesInputItemTokenOverhead
		if err := countText(item.Role); err != nil {
			return 0, err
		}
		if item.Type != "" && item.Type != "message" {
			if err := countText(item.Type); err != nil {
				return 0, err
			}
		}
		if err := countText(item.Name); err != nil {
			return 0, err
		}
		if err := countText(item.Arguments); err != nil {
			return 0, err
		}
		if err := countText(item.Output); err != nil {
			return 0, err
		}
		if err := countText(item.CallID); err != nil {
			return 0, err
		}
		if err := countText(item.ID); err != nil {
			return 0, err
		}

		if len(bytes.TrimSpace(item.Content)) == 0 {
			continue
		}

		var contentText string
		if err := json.Unmarshal(item.Content, &contentText); err == nil {
			if err := countText(contentText); err != nil {
				return 0, err
			}
			continue
		}

		var parts []apicompat.ResponsesContentPart
		if err := json.Unmarshal(item.Content, &parts); err == nil {
			for _, part := range parts {
				total += openAIResponsesContentPartOverhead
				switch part.Type {
				case "input_text", "output_text", "text":
					if err := countText(part.Text); err != nil {
						return 0, err
					}
				case "input_image":
					if err := countText(estimateOpenAIInputImageText(part.ImageURL)); err != nil {
						return 0, err
					}
				default:
					if err := countText(part.Type); err != nil {
						return 0, err
					}
				}
			}
			continue
		}

		compacted, err := compactOpenAIInputTokensJSON(item.Content)
		if err != nil {
			return 0, err
		}
		if err := countText(compacted); err != nil {
			return 0, err
		}
	}

	return total, nil
}

func estimateOpenAIInputImageText(imageURL string) string {
	trimmed := strings.TrimSpace(imageURL)
	if trimmed == "" {
		return ""
	}
	if strings.HasPrefix(strings.ToLower(trimmed), "data:") {
		if comma := strings.Index(trimmed, ","); comma > 0 {
			return trimmed[:comma]
		}
	}
	return trimmed
}

func compactOpenAIInputTokensJSON(raw json.RawMessage) (string, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return "", nil
	}
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		return "", err
	}
	return buf.String(), nil
}

func openAIInputTokensCodecForModel(model string) (tokenizer.Codec, error) {
	switch openAIInputTokensEncodingForModel(model) {
	case tokenizer.Cl100kBase:
		return tokenizer.Get(tokenizer.Cl100kBase)
	default:
		return tokenizer.Get(tokenizer.O200kBase)
	}
}

func openAIInputTokensEncodingForModel(model string) tokenizer.Encoding {
	normalized := strings.ToLower(strings.TrimSpace(model))
	switch {
	case strings.HasPrefix(normalized, "gpt-3.5"),
		(strings.HasPrefix(normalized, "gpt-4") &&
			!strings.HasPrefix(normalized, "gpt-4o") &&
			!strings.HasPrefix(normalized, "gpt-4.1")),
		strings.HasPrefix(normalized, "text-embedding-"):
		return tokenizer.Cl100kBase
	default:
		return tokenizer.O200kBase
	}
}
