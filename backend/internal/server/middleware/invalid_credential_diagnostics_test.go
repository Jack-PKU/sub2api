package middleware

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zapcore"
)

func diagnosticFieldMap(t *testing.T, headers map[string]string) map[string]any {
	t.Helper()
	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	for key, value := range headers {
		c.Request.Header.Set(key, value)
	}
	encoder := zapcore.NewMapObjectEncoder()
	for _, field := range invalidCredentialFields(c) {
		field.AddTo(encoder)
	}
	return encoder.Fields
}

func TestInvalidCredentialFieldsNeverLogTheCredential(t *testing.T) {
	secret := "sk-ant-oat01-" + strings.Repeat("a", 90)
	fields := diagnosticFieldMap(t, map[string]string{
		"Authorization":            "Bearer " + secret,
		"User-Agent":               "claude-cli/2.1.284 (external, claude-desktop)",
		"X-Claude-Code-Session-Id": "bfdeffff-402a-487f-ae02-61906597ff3d",
	})
	require.Equal(t, "anthropic_oauth_token", fields["credential_class"])
	require.Equal(t, "authorization_bearer", fields["credential_source"])
	require.Equal(t, int64(len(secret)), fields["credential_length"])
	require.Len(t, fields["credential_sha256_12"], 12)
	require.Equal(t, "bfdeffff-402a-487f-ae02-61906597ff3d", fields["x_claude_code_session_id"])
	for _, value := range fields {
		if text, ok := value.(string); ok {
			require.NotContains(t, text, "aaaaaaaaaa")
		}
	}
}

func TestCredentialClassFamilies(t *testing.T) {
	require.Equal(t, "empty", credentialClass(""))
	require.Equal(t, "anthropic_api_key", credentialClass("sk-ant-api03-xyz"))
	require.Equal(t, "sk_prefixed_67", credentialClass("sk-"+strings.Repeat("0", 64)))
	require.Equal(t, "jwt", credentialClass("eyJhbGciOi"))
	require.Equal(t, "other", credentialClass("plain"))
}

func TestInvalidCredentialFieldsXAPIKeyAndTruncatedUserAgent(t *testing.T) {
	fields := diagnosticFieldMap(t, map[string]string{
		"x-api-key":  "sk-" + strings.Repeat("1", 64),
		"User-Agent": strings.Repeat("u", 400),
	})
	require.Equal(t, "x_api_key", fields["credential_source"])
	require.Len(t, fields["user_agent"], diagnosticUserAgentMaxBytes)
	_, hasSession := fields["x_claude_code_session_id"]
	require.False(t, hasSession)
}
