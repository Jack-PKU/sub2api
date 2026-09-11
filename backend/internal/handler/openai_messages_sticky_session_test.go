package handler

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newStickyTestContext(t *testing.T, headers map[string]string) *gin.Context {
	t.Helper()
	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	for name, value := range headers {
		req.Header.Set(name, value)
	}
	c.Request = req
	return c
}

// The defect this fixes: a coding harness rewrites its request body every turn,
// so the body-derived content seed differed on every request, a fresh sticky key
// was written each time, the pin never held, and account selection fell through
// to LRU and alternated upstream accounts mid-conversation. The stable client
// session id must produce one identical sticky hash across turns.
func TestClientStickySessionIsStableAcrossTurns(t *testing.T) {
	const sessionID = "ba881f4b-019a-4ae0-b09d-92b9fec53023"

	turn1 := newStickyTestContext(t, map[string]string{"X-Claude-Code-Session-Id": sessionID})
	turn2 := newStickyTestContext(t, map[string]string{"X-Claude-Code-Session-Id": sessionID})

	// Distinct per-turn content seeds, as the real client produces.
	hash1 := resolveOpenAIMessagesClientStickySession(turn1, "content-seed-turn-1", "claude-opus-5", nil)
	hash2 := resolveOpenAIMessagesClientStickySession(turn2, "content-seed-turn-2", "claude-opus-5", nil)

	require.NotEmpty(t, hash1)
	assert.Equal(t, hash1, hash2, "same conversation must keep one sticky key across turns")
	assert.NotEqual(t, "content-seed-turn-1", hash1, "content seed must be replaced when a session id is present")
}

// Toggling the 1M-context selector mid-conversation must not break the pin,
// because the seed uses the normalized routing model.
func TestClientStickySessionIgnoresSelectorSuffix(t *testing.T) {
	const sessionID = "session-abc"
	c := newStickyTestContext(t, map[string]string{"X-Claude-Code-Session-Id": sessionID})

	base := resolveOpenAIMessagesClientStickySession(c, "seed-a", "claude-opus-5", nil)
	long := resolveOpenAIMessagesClientStickySession(c, "seed-b", "claude-opus-5[1m]", nil)

	assert.Equal(t, base, long, "base and 1M selectors normalize to one sticky key")
}

// Different conversations must not collide onto one account pin.
func TestClientStickySessionSeparatesConversations(t *testing.T) {
	a := newStickyTestContext(t, map[string]string{"X-Claude-Code-Session-Id": "session-a"})
	b := newStickyTestContext(t, map[string]string{"X-Claude-Code-Session-Id": "session-b"})

	assert.NotEqual(t,
		resolveOpenAIMessagesClientStickySession(a, "seed", "claude-opus-5", nil),
		resolveOpenAIMessagesClientStickySession(b, "seed", "claude-opus-5", nil))
}

// Different models in the same client session stay separate, matching the
// existing metadata.user_id seeding which also includes the model.
func TestClientStickySessionSeparatesModels(t *testing.T) {
	c := newStickyTestContext(t, map[string]string{"X-Claude-Code-Session-Id": "session-a"})

	assert.NotEqual(t,
		resolveOpenAIMessagesClientStickySession(c, "seed", "claude-opus-5", nil),
		resolveOpenAIMessagesClientStickySession(c, "seed", "claude-sonnet-5", nil))
}

// A caller that sends no session header keeps the previous content-seed
// behaviour byte-for-byte; this change must not alter non-harness traffic.
func TestClientStickySessionFallsBackToContentSeed(t *testing.T) {
	c := newStickyTestContext(t, nil)

	assert.Equal(t, "content-seed", resolveOpenAIMessagesClientStickySession(c, "content-seed", "claude-opus-5", nil))
	assert.Equal(t, "", resolveOpenAIMessagesClientStickySession(c, "", "claude-opus-5", nil))
}

// A header that the sanitizer rejects (control characters, over-long) must not
// silently produce a degenerate pin; fall back to the content seed instead.
func TestClientStickySessionRejectsUnsafeHeader(t *testing.T) {
	for name, value := range map[string]string{
		"control character": "abc\ndef",
		"whitespace only":   "   ",
	} {
		t.Run(name, func(t *testing.T) {
			c := newStickyTestContext(t, map[string]string{"X-Claude-Code-Session-Id": value})
			assert.Equal(t, "content-seed",
				resolveOpenAIMessagesClientStickySession(c, "content-seed", "claude-opus-5", nil))
		})
	}
}

// The sticky hash must be derived through the same helper the scheduler uses to
// read and write bindings, or a pin would be written under a key that is never
// looked up.
func TestClientStickySessionUsesSchedulerDerivation(t *testing.T) {
	c := newStickyTestContext(t, map[string]string{"X-Claude-Code-Session-Id": "session-x"})

	assert.Equal(t,
		service.DeriveSessionHashFromSeed(service.NormalizeStickySessionModel("claude-opus-5")+"-session-x"),
		resolveOpenAIMessagesClientStickySession(c, "seed", "claude-opus-5", nil))
}
