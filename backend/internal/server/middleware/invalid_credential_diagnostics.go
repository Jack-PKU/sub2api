package middleware

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

const diagnosticUserAgentMaxBytes = 160

// invalidCredentialFields describes the client and the shape of a rejected
// credential so a shared-IP caller sending a wrong key can be traced (all local
// Claude Code seats reach the gateway from one docker bridge address). Only a
// non-reversible fingerprint, the length and a coarse prefix class of the
// credential are logged - never the credential itself.
func invalidCredentialFields(c *gin.Context) []zap.Field {
	credential, source := presentedCredential(c)
	fields := []zap.Field{
		zap.String("user_agent", truncateForLog(c.GetHeader("User-Agent"), diagnosticUserAgentMaxBytes)),
		zap.String("credential_source", source),
		zap.String("credential_class", credentialClass(credential)),
		zap.Int("credential_length", len(credential)),
	}
	if credential != "" {
		digest := sha256.Sum256([]byte(credential))
		fields = append(fields, zap.String("credential_sha256_12", hex.EncodeToString(digest[:])[:12]))
	}
	for _, header := range []string{"X-Claude-Code-Session-Id", "X-App", "X-Claude-Code-Agent-Id"} {
		if value := strings.TrimSpace(c.GetHeader(header)); value != "" {
			fields = append(fields, zap.String(strings.ToLower(strings.ReplaceAll(header, "-", "_")), truncateForLog(value, 80)))
		}
	}
	return fields
}

func presentedCredential(c *gin.Context) (string, string) {
	if header := c.GetHeader("Authorization"); header != "" {
		parts := strings.SplitN(header, " ", 2)
		if len(parts) == 2 && strings.EqualFold(parts[0], "Bearer") {
			return strings.TrimSpace(parts[1]), "authorization_bearer"
		}
		return "", "authorization_other"
	}
	if key := c.GetHeader("x-api-key"); key != "" {
		return key, "x_api_key"
	}
	if key := c.GetHeader("x-goog-api-key"); key != "" {
		return key, "x_goog_api_key"
	}
	return "", "none"
}

// credentialClass names the family a credential belongs to without revealing it.
func credentialClass(credential string) string {
	switch {
	case credential == "":
		return "empty"
	case strings.HasPrefix(credential, "sk-ant-oat"):
		return "anthropic_oauth_token"
	case strings.HasPrefix(credential, "sk-ant-"):
		return "anthropic_api_key"
	case strings.HasPrefix(credential, "sk-"):
		return "sk_prefixed_" + strconv.Itoa(len(credential))
	case strings.HasPrefix(credential, "eyJ"):
		return "jwt"
	default:
		return "other"
	}
}

func truncateForLog(value string, limit int) string {
	if len(value) <= limit {
		return value
	}
	return value[:limit]
}
