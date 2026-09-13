package service

import (
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// claim is a readability helper: the release is irrelevant to most assertions.
func claim(s *OpenAIGatewayService, account *Account) bool {
	serveLocally, release := s.claimOpenAIInputTokensProbe(account)
	release()
	return serveLocally
}

// A ChatGPT-subscription OAuth credential cannot reach the platform
// input_tokens endpoint, so after the first proof the probe must be skipped.
func TestInputTokensProbeSkippedAfterUnsupportedVerdict(t *testing.T) {
	s := &OpenAIGatewayService{}
	account := &Account{ID: 42, Type: AccountTypeOAuth}

	assert.False(t, claim(s, account), "first call must probe upstream")

	s.rememberOpenAIInputTokensUnsupported(account, 401)
	assert.True(t, claim(s, account), "verdict must suppress the next probe")
}

// The regression this exists for: a client fires count_tokens in bursts of 80+
// parallel calls, and a verdict recorded only on completion is still empty when
// the whole burst reads it. Exactly one caller may probe; the rest answer
// locally rather than queueing behind a round trip that is about to fail.
func TestConcurrentBurstCollapsesToOneProbe(t *testing.T) {
	s := &OpenAIGatewayService{}
	account := &Account{ID: 1, Type: AccountTypeOAuth}

	const burst = 80
	var probes int64
	var mu sync.Mutex
	var wg sync.WaitGroup
	start := make(chan struct{})
	releases := make([]func(), 0, burst)

	for i := 0; i < burst; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			serveLocally, release := s.claimOpenAIInputTokensProbe(account)
			mu.Lock()
			defer mu.Unlock()
			if !serveLocally {
				probes++
				// Hold the claim, as a real in-flight probe would.
				releases = append(releases, release)
			} else {
				release()
			}
		}()
	}
	close(start)
	wg.Wait()

	assert.Equal(t, int64(1), probes, "a parallel burst must produce exactly one upstream probe")
	for _, release := range releases {
		release()
	}
}

// After the single probe records its verdict, the whole next burst is served
// locally with no probe at all.
func TestBurstAfterVerdictProbesNothing(t *testing.T) {
	s := &OpenAIGatewayService{}
	account := &Account{ID: 2, Type: AccountTypeOAuth}
	s.rememberOpenAIInputTokensUnsupported(account, 401)

	for i := 0; i < 50; i++ {
		assert.True(t, claim(s, account))
	}
}

// An endpoint that actually answers must keep answering: a supported account
// never skips, so exact upstream counts are not silently downgraded to local
// estimates after the first burst.
func TestSupportedAccountNeverSkipsProbe(t *testing.T) {
	s := &OpenAIGatewayService{}
	account := &Account{ID: 3, Type: AccountTypeOAuth}

	s.rememberOpenAIInputTokensSupported(account)

	first, releaseFirst := s.claimOpenAIInputTokensProbe(account)
	second, releaseSecond := s.claimOpenAIInputTokensProbe(account)
	releaseFirst()
	releaseSecond()

	assert.False(t, first)
	assert.False(t, second, "a supported account must not collapse concurrent callers")
}

// The verdict is per account: one credential's missing scope must not downgrade
// another that may still have it.
func TestInputTokensVerdictIsPerAccount(t *testing.T) {
	s := &OpenAIGatewayService{}
	blocked := &Account{ID: 1, Type: AccountTypeOAuth}
	other := &Account{ID: 2, Type: AccountTypeOAuth}

	s.rememberOpenAIInputTokensUnsupported(blocked, 401)

	assert.True(t, claim(s, blocked))
	assert.False(t, claim(s, other))
}

// API-key accounts reach the endpoint; suppressing or collapsing their probes
// would replace an exact upstream count with a local estimate.
func TestNonOAuthAccountIsNeverCollapsed(t *testing.T) {
	s := &OpenAIGatewayService{}
	apiKeyAccount := &Account{ID: 7, Type: AccountTypeAPIKey}

	s.rememberOpenAIInputTokensUnsupported(apiKeyAccount, 401)
	s.rememberOpenAIInputTokensSupported(apiKeyAccount)

	_, stored := s.openaiInputTokensProbeState.Load(apiKeyAccount.ID)
	assert.False(t, stored, "non-OAuth accounts must not be recorded")

	first, releaseFirst := s.claimOpenAIInputTokensProbe(apiKeyAccount)
	second, releaseSecond := s.claimOpenAIInputTokensProbe(apiKeyAccount)
	releaseFirst()
	releaseSecond()
	assert.False(t, first)
	assert.False(t, second)
}

// An expired verdict must resume probing so a re-authorized credential recovers
// without a gateway restart, and the stale entry must be replaced.
func TestExpiredVerdictResumesProbing(t *testing.T) {
	s := &OpenAIGatewayService{}
	account := &Account{ID: 9, Type: AccountTypeOAuth}

	s.openaiInputTokensProbeState.Store(account.ID, time.Now().Add(-time.Minute))

	serveLocally, release := s.claimOpenAIInputTokensProbe(account)
	assert.False(t, serveLocally, "expired verdict must not suppress")
	state, stored := s.openaiInputTokensProbeState.Load(account.ID)
	require.True(t, stored)
	assert.Equal(t, openAIInputTokensProbeRunning, state, "the re-probe must hold the claim")
	release()
}

// A probe that ends without recording a verdict (network error) must release the
// claim, or the account would be wedged into "serve locally" forever.
func TestFailedProbeReleasesTheClaim(t *testing.T) {
	s := &OpenAIGatewayService{}
	account := &Account{ID: 10, Type: AccountTypeOAuth}

	serveLocally, release := s.claimOpenAIInputTokensProbe(account)
	require.False(t, serveLocally)
	assert.True(t, claim(s, account), "a second caller waits behind the in-flight probe")

	release()

	assert.False(t, claim(s, account), "after the probe ends the next caller may probe again")
}

// Releasing after a verdict was recorded must not erase it.
func TestReleaseDoesNotClobberARecordedVerdict(t *testing.T) {
	s := &OpenAIGatewayService{}
	account := &Account{ID: 11, Type: AccountTypeOAuth}

	serveLocally, release := s.claimOpenAIInputTokensProbe(account)
	require.False(t, serveLocally)
	s.rememberOpenAIInputTokensUnsupported(account, 401)
	release()

	assert.True(t, claim(s, account), "the verdict must survive the claim release")
}

// A corrupt cache entry must fail open (probe upstream) rather than panic or
// suppress forever.
func TestCorruptStateFailsOpen(t *testing.T) {
	s := &OpenAIGatewayService{}
	account := &Account{ID: 12, Type: AccountTypeOAuth}

	s.openaiInputTokensProbeState.Store(account.ID, "not-a-verdict")

	serveLocally, release := s.claimOpenAIInputTokensProbe(account)
	release()
	assert.False(t, serveLocally)
}

func TestClaimHandlesNilAccount(t *testing.T) {
	s := &OpenAIGatewayService{}
	assert.NotPanics(t, func() {
		serveLocally, release := s.claimOpenAIInputTokensProbe(nil)
		release()
		assert.False(t, serveLocally)
		s.rememberOpenAIInputTokensUnsupported(nil, 401)
		s.rememberOpenAIInputTokensSupported(nil)
	})
}

// The TTL must bound the verdict to a real window: long enough that a session
// costs at most one probe, short enough that a credential fix is picked up the
// same day.
func TestUnsupportedTTLIsBounded(t *testing.T) {
	assert.GreaterOrEqual(t, openAIInputTokensUnsupportedTTL, time.Hour)
	assert.LessOrEqual(t, openAIInputTokensUnsupportedTTL, 24*time.Hour)

	s := &OpenAIGatewayService{}
	account := &Account{ID: 4, Type: AccountTypeOAuth}
	before := time.Now()
	s.rememberOpenAIInputTokensUnsupported(account, 401)

	raw, ok := s.openaiInputTokensProbeState.Load(account.ID)
	require.True(t, ok)
	until, ok := raw.(time.Time)
	require.True(t, ok)
	assert.WithinDuration(t, before.Add(openAIInputTokensUnsupportedTTL), until, time.Minute)
}
