package service

import (
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// anthropicOverflowTriggerMatches mirrors how an Anthropic Messages client
// decides a rejection is a recoverable context overflow: a lowercase substring
// test for one of two marker phrases. If this does not match, the client shows a
// dead end instead of compacting and retrying.
func anthropicOverflowTriggerMatches(message string) bool {
	lower := strings.ToLower(message)
	return strings.Contains(lower, "prompt is too long") ||
		strings.Contains(lower, "input is too long for requested model")
}

func TestNormalizeAnthropicContextWindowMessage_RewritesOpenAIWording(t *testing.T) {
	cases := []struct {
		name     string
		upstream string
	}{
		{"context window prose", "Your input exceeds the context window of this model. Please adjust your input and try again."},
		{"error code", "context_length_exceeded"},
		{"max context length", "This model's maximum context length is 922000 tokens."},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			require.False(t, anthropicOverflowTriggerMatches(tc.upstream),
				"precondition: raw upstream wording must not already trigger the client")

			got := normalizeAnthropicContextWindowMessage(http.StatusBadRequest, "invalid_request_error", tc.upstream)

			assert.True(t, anthropicOverflowTriggerMatches(got), "rewritten message must trigger client compaction: %q", got)
			assert.Contains(t, got, tc.upstream, "original upstream text must be preserved for diagnosis")
		})
	}
}

// When the upstream reports both numbers, the rewritten message must carry them
// in the exact "<used> tokens > <limit>" shape, with no other digits between the
// marker phrase and the used count.
func TestNormalizeAnthropicContextWindowMessage_CarriesTokenCounts(t *testing.T) {
	upstream := "This model's maximum context length is 922000 tokens. However, your messages resulted in 930500 tokens. Please reduce the length of the messages."

	got := normalizeAnthropicContextWindowMessage(http.StatusBadRequest, "invalid_request_error", upstream)

	assert.True(t, strings.HasPrefix(got, "prompt is too long: 930500 tokens > 922000 maximum"), "got %q", got)
	assert.Contains(t, got, upstream)

	used, limit, ok := openAIContextWindowTokenCounts(upstream)
	require.True(t, ok)
	assert.Equal(t, 930500, used)
	assert.Equal(t, 922000, limit)
}

// Inconsistent or partial numbers must degrade to the count-free form rather
// than emitting invented or reversed counts.
func TestOpenAIContextWindowTokenCounts_RejectsInconsistent(t *testing.T) {
	cases := []string{
		"This model's maximum context length is 922000 tokens.",                                   // no used count
		"However, your messages resulted in 930500 tokens.",                                       // no limit
		"maximum context length is 922000 tokens. However, your messages resulted in 900 tokens.", // used < limit
		"maximum context length is 0 tokens. However, your messages resulted in 100 tokens.",      // zero limit
	}
	for _, upstream := range cases {
		_, _, ok := openAIContextWindowTokenCounts(upstream)
		assert.False(t, ok, "must not extract counts from %q", upstream)
	}

	got := normalizeAnthropicContextWindowMessage(http.StatusBadRequest, "invalid_request_error", cases[0])
	assert.True(t, anthropicOverflowTriggerMatches(got))
	assert.NotContains(t, got, "tokens > ", "count-free form must not fabricate a comparison")
}

// Anything that is not an invalid-request context overflow must pass through
// byte-identical: rate limits, auth failures, server errors, and unrelated
// invalid-request rejections.
func TestNormalizeAnthropicContextWindowMessage_LeavesOtherErrorsUntouched(t *testing.T) {
	cases := []struct {
		name    string
		status  int
		errType string
		message string
	}{
		{"rate limit", http.StatusTooManyRequests, "rate_limit_error", "Upstream rate limit exceeded, please retry later"},
		{"auth", http.StatusUnauthorized, "authentication_error", "Invalid credentials"},
		{"server", http.StatusBadGateway, "api_error", "Upstream request failed"},
		{"unrelated invalid request", http.StatusBadRequest, "invalid_request_error", "Unsupported parameter: 'top_k'"},
		{"overflow but wrong type", http.StatusBadRequest, "api_error", "context_length_exceeded"},
		{"overflow but wrong status", http.StatusInternalServerError, "invalid_request_error", "context_length_exceeded"},
		{"empty", http.StatusBadRequest, "invalid_request_error", ""},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.message,
				normalizeAnthropicContextWindowMessage(tc.status, tc.errType, tc.message))
		})
	}
}

// A message already in the expected wording must not be wrapped twice.
func TestNormalizeAnthropicContextWindowMessage_Idempotent(t *testing.T) {
	first := normalizeAnthropicContextWindowMessage(http.StatusBadRequest, "invalid_request_error",
		"maximum context length is 922000 tokens. However, your messages resulted in 930500 tokens.")
	second := normalizeAnthropicContextWindowMessage(http.StatusBadRequest, "invalid_request_error", first)
	assert.Equal(t, first, second)

	native := "prompt is too long: 300000 tokens > 200000 maximum"
	assert.Equal(t, native,
		normalizeAnthropicContextWindowMessage(http.StatusBadRequest, "invalid_request_error", native))
}

// 413 carries the same overflow semantics on some upstreams and must be
// rewritten too.
func TestNormalizeAnthropicContextWindowMessage_RequestEntityTooLarge(t *testing.T) {
	got := normalizeAnthropicContextWindowMessage(http.StatusRequestEntityTooLarge, "invalid_request_error",
		"Your input exceeds the context window of this model.")
	assert.True(t, anthropicOverflowTriggerMatches(got), "got %q", got)
}
