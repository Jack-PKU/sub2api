package service

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

// Mirrors the tail of a real Claude Code 2.1.284 /compact request: the
// instruction is its own text block in the last user turn, followed by a
// mid-conversation system message.
const claudeCodeCompactionMessagesJSON = `[
	{"role":"user","content":"Summarize the repo"},
	{"role":"assistant","content":[{"type":"text","text":"PONG"}]},
	{"role":"user","content":[
		{"type":"text","text":"<system-reminder>context</system-reminder>"},
		{"type":"text","text":"<command-name>/compact</command-name>"},
		{"type":"text","text":"CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\n- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.\n\nYour task is to create a detailed summary of the conversation so far."}
	]},
	{"role":"system","content":"Today's date is 2026-09-30."}
]`

func anthropicRequestWithMessages(t *testing.T, messagesJSON string) *apicompat.AnthropicRequest {
	t.Helper()
	req := &apicompat.AnthropicRequest{}
	require.NoError(t, json.Unmarshal([]byte(messagesJSON), &req.Messages))
	return req
}

func TestCapReasoningEffort(t *testing.T) {
	tests := []struct{ effort, ceiling, want string }{
		{"max", "medium", "medium"},
		{"xhigh", "high", "high"},
		{"high", "high", "high"},
		{"low", "medium", "low"},
		{"max", "", "max"},
		{"none", "medium", "none"},
		{"", "medium", ""},
	}
	for _, tt := range tests {
		require.Equal(t, tt.want, capReasoningEffort(tt.effort, tt.ceiling), "effort=%q ceiling=%q", tt.effort, tt.ceiling)
	}
}

func TestIsClaudeCodeCompactionRequest(t *testing.T) {
	marker := claudeCodeCompactionInstructionPrefix
	tests := []struct {
		name     string
		messages string
		want     bool
	}{
		{name: "manual compact with trailing system message", messages: claudeCodeCompactionMessagesJSON, want: true},
		{
			name:     "auto compact merged after tool results",
			messages: `[{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Bash","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"},{"type":"text","text":"\n` + marker + `\nYour task is to create a detailed summary"}]}]`,
			want:     true,
		},
		{name: "string content", messages: `[{"role":"user","content":"` + marker + ` Summarize."}]`, want: true},
		{name: "ordinary turn", messages: `[{"role":"user","content":"fix the failing test"}]`, want: false},
		{
			name:     "tool result quoting the instruction",
			messages: `[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"` + marker + `"}]}]`,
			want:     false,
		},
		{
			name:     "file mention quoting the instruction mid-text",
			messages: `[{"role":"user","content":[{"type":"text","text":"Contents of effort.go: const prefix = \"` + marker + `\""}]}]`,
			want:     false,
		},
		{
			name:     "instruction only in an earlier user turn",
			messages: `[{"role":"user","content":"` + marker + `"},{"role":"assistant","content":"summary"},{"role":"user","content":"continue"}]`,
			want:     false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, isClaudeCodeCompactionRequest(anthropicRequestWithMessages(t, tt.messages)))
		})
	}
}

func TestForwardAsAnthropic_CapsClaudeCodeCompactionAndSubagentEffort(t *testing.T) {
	gin.SetMode(gin.TestMode)
	ordinaryTurn := `[{"role":"user","content":"hello"}]`
	tests := []struct {
		name       string
		messages   string
		agentID    string
		effort     string
		wantEffort string
	}{
		{name: "main thread keeps max", messages: ordinaryTurn, effort: "max", wantEffort: "max"},
		{name: "subagent capped at high", messages: ordinaryTurn, agentID: "a2ef546cb3fc3480c", effort: "max", wantEffort: "high"},
		{name: "subagent below the cap unchanged", messages: ordinaryTurn, agentID: "a2ef546cb3fc3480c", effort: "low", wantEffort: "low"},
		{name: "compaction capped at medium", messages: claudeCodeCompactionMessagesJSON, effort: "max", wantEffort: "medium"},
		{name: "subagent compaction uses the lower cap", messages: claudeCodeCompactionMessagesJSON, agentID: "a2ef546cb3fc3480c", effort: "max", wantEffort: "medium"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := `{"model":"sol","max_tokens":64,"messages":` + tt.messages + `,"thinking":{"type":"adaptive"},"output_config":{"effort":"` + tt.effort + `"},"stream":false}`
			rec := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(rec)
			c.Request = httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
			c.Request.Header.Set("Content-Type", "application/json")
			if tt.agentID != "" {
				c.Request.Header.Set(claudeCodeAgentIDHeader, tt.agentID)
			}

			upstream := &httpUpstreamRecorder{resp: openAICompatSSECompletedResponse("resp_cc_effort", "gpt-5.6-sol")}
			svc := &OpenAIGatewayService{
				httpUpstream: upstream,
				cfg:          &config.Config{Security: config.SecurityConfig{URLAllowlist: config.URLAllowlistConfig{Enabled: false}}},
			}

			result, err := svc.ForwardAsAnthropic(context.Background(), c, rawGPT56ResponsesOAuthAccount("sol", "gpt-5.6-sol"), []byte(body), "", "")
			require.NoError(t, err)
			require.NotNil(t, result)
			require.Equal(t, tt.wantEffort, gjson.GetBytes(upstream.lastBody, "reasoning.effort").String())
			require.NotNil(t, result.ReasoningEffort)
			require.Equal(t, tt.wantEffort, *result.ReasoningEffort)
		})
	}
}
