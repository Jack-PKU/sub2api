package service

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A ChatGPT-subscription OAuth credential cannot reach the platform
// input_tokens endpoint, so after the first proof the probe must be skipped.
func TestOpenAIInputTokensProbeSuppressed_AfterOAuthVerdict(t *testing.T) {
	s := &OpenAIGatewayService{}
	account := &Account{ID: 42, Type: AccountTypeOAuth}

	assert.False(t, s.openAIInputTokensProbeSuppressed(account), "first call must probe upstream")

	s.rememberOpenAIInputTokensUnsupported(account, 401)
	assert.True(t, s.openAIInputTokensProbeSuppressed(account), "verdict must suppress the next probe")
}

// The verdict is per account: one account's missing scope must not downgrade
// another account that may still have it.
func TestOpenAIInputTokensProbeSuppressed_IsPerAccount(t *testing.T) {
	s := &OpenAIGatewayService{}
	blocked := &Account{ID: 1, Type: AccountTypeOAuth}
	other := &Account{ID: 2, Type: AccountTypeOAuth}

	s.rememberOpenAIInputTokensUnsupported(blocked, 401)

	assert.True(t, s.openAIInputTokensProbeSuppressed(blocked))
	assert.False(t, s.openAIInputTokensProbeSuppressed(other))
}

// API-key accounts do reach the endpoint; suppressing their probe would replace
// an exact upstream count with a local estimate.
func TestOpenAIInputTokensProbe_NotSuppressedForNonOAuth(t *testing.T) {
	s := &OpenAIGatewayService{}
	apiKeyAccount := &Account{ID: 7, Type: AccountTypeAPIKey}

	s.rememberOpenAIInputTokensUnsupported(apiKeyAccount, 401)

	_, stored := s.openaiInputTokensUnsupportedUntil.Load(apiKeyAccount.ID)
	assert.False(t, stored, "non-OAuth accounts must not be memoized")
	assert.False(t, s.openAIInputTokensProbeSuppressed(apiKeyAccount))
}

// An expired verdict must resume probing so a re-authorized credential recovers
// without a gateway restart, and the stale entry must be dropped.
func TestOpenAIInputTokensProbeSuppressed_ExpiresAndClears(t *testing.T) {
	s := &OpenAIGatewayService{}
	account := &Account{ID: 9, Type: AccountTypeOAuth}

	s.openaiInputTokensUnsupportedUntil.Store(account.ID, time.Now().Add(-time.Minute))

	assert.False(t, s.openAIInputTokensProbeSuppressed(account), "expired verdict must not suppress")
	_, stored := s.openaiInputTokensUnsupportedUntil.Load(account.ID)
	assert.False(t, stored, "expired verdict must be removed")
}

// A corrupt cache entry must fail open (probe upstream) rather than panic or
// suppress forever.
func TestOpenAIInputTokensProbeSuppressed_IgnoresCorruptEntry(t *testing.T) {
	s := &OpenAIGatewayService{}
	account := &Account{ID: 11, Type: AccountTypeOAuth}

	s.openaiInputTokensUnsupportedUntil.Store(account.ID, "not-a-time")

	assert.False(t, s.openAIInputTokensProbeSuppressed(account))
	_, stored := s.openaiInputTokensUnsupportedUntil.Load(account.ID)
	assert.False(t, stored)
}

func TestOpenAIInputTokensProbeSuppressed_NilAccount(t *testing.T) {
	s := &OpenAIGatewayService{}
	assert.NotPanics(t, func() {
		assert.False(t, s.openAIInputTokensProbeSuppressed(nil))
		s.rememberOpenAIInputTokensUnsupported(nil, 401)
	})
}

// The TTL must bound the verdict to a real window: long enough that one session
// (including its end-of-turn burst) costs at most one probe, short enough that a
// credential fix is picked up the same day.
func TestOpenAIInputTokensUnsupportedTTLIsBounded(t *testing.T) {
	assert.GreaterOrEqual(t, openAIInputTokensUnsupportedTTL, time.Hour)
	assert.LessOrEqual(t, openAIInputTokensUnsupportedTTL, 24*time.Hour)

	s := &OpenAIGatewayService{}
	account := &Account{ID: 3, Type: AccountTypeOAuth}
	before := time.Now()
	s.rememberOpenAIInputTokensUnsupported(account, 401)

	raw, ok := s.openaiInputTokensUnsupportedUntil.Load(account.ID)
	require.True(t, ok)
	until, ok := raw.(time.Time)
	require.True(t, ok)
	assert.WithinDuration(t, before.Add(openAIInputTokensUnsupportedTTL), until, time.Minute)
}
