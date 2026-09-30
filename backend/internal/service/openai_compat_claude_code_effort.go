package service

import (
	"encoding/json"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
	"github.com/gin-gonic/gin"
)

// Claude Code sends every request of a session at the effort chosen for that
// session, usually max. Two request kinds pay for it on ChatGPT/Grok
// subscriptions without needing it: compaction, which rewrites the whole
// context as a summary, and turns of subagents the main thread delegated to.
// At max they routinely run 10-15 minutes and reach the upstream's ~15 minute
// stream cut, so they are capped here while the main thread keeps its effort.
const (
	claudeCodeCompactionEffortCeiling = "medium"
	claudeCodeSubagentEffortCeiling   = "high"

	// Sent only on requests made by a subagent (Agent tool or workflow agent);
	// the main thread omits it.
	claudeCodeAgentIDHeader = "X-Claude-Code-Agent-Id"

	// Claude Code opens its compaction instruction, for both the full and the
	// partial summary, with this sentence and uses it nowhere else. Its
	// request-class headers are only sent to Anthropic's own endpoint, so the
	// instruction itself is the signal a gateway can see.
	claudeCodeCompactionInstructionPrefix = "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools."
)

// claudeCodeEffortCeiling returns the highest reasoning effort a Claude Code
// request should run at, or "" when the client's choice stands.
func claudeCodeEffortCeiling(c *gin.Context, req *apicompat.AnthropicRequest) string {
	if isClaudeCodeCompactionRequest(req) {
		return claudeCodeCompactionEffortCeiling
	}
	if claudeCodeAgentID(c) != "" {
		return claudeCodeSubagentEffortCeiling
	}
	return ""
}

// capReasoningEffort lowers effort to ceiling. Values outside the ranked scale,
// such as "none" or an absent effort, keep their meaning.
func capReasoningEffort(effort, ceiling string) string {
	current, ok := reasoningEffortRank(effort)
	if !ok {
		return effort
	}
	limit, ok := reasoningEffortRank(ceiling)
	if !ok || current <= limit {
		return effort
	}
	return ceiling
}

func claudeCodeAgentID(c *gin.Context) string {
	if c == nil || c.Request == nil {
		return ""
	}
	return strings.TrimSpace(c.GetHeader(claudeCodeAgentIDHeader))
}

// isClaudeCodeCompactionRequest inspects the last user turn, where the client
// appends the instruction as its own text block, possibly after tool results
// and before a trailing mid-conversation system message. Only a block that
// starts with the instruction counts, so a file quoting it (an @-mention or a
// tool result) does not.
func isClaudeCodeCompactionRequest(req *apicompat.AnthropicRequest) bool {
	if req == nil {
		return false
	}
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == "user" {
			return anthropicContentHasTextBlockPrefix(req.Messages[i].Content, claudeCodeCompactionInstructionPrefix)
		}
	}
	return false
}

func anthropicContentHasTextBlockPrefix(content json.RawMessage, prefix string) bool {
	var text string
	if err := json.Unmarshal(content, &text); err == nil {
		return strings.HasPrefix(strings.TrimSpace(text), prefix)
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(content, &blocks); err != nil {
		return false
	}
	for _, block := range blocks {
		if block.Type == "text" && strings.HasPrefix(strings.TrimSpace(block.Text), prefix) {
			return true
		}
	}
	return false
}
