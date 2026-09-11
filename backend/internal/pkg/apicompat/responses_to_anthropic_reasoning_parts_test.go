package apicompat

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// collectThinking concatenates every thinking_delta the converter emitted for a
// stream, which is exactly what a client renders into one thinking block.
func collectThinking(events []AnthropicStreamEvent) string {
	var b strings.Builder
	for _, ev := range events {
		if ev.Type == "content_block_delta" && ev.Delta != nil && ev.Delta.Type == "thinking_delta" {
			b.WriteString(ev.Delta.Thinking)
		}
	}
	return b.String()
}

// A reasoning item split into three summary parts must render as three
// paragraphs. Before the fix the parts were concatenated bare and produced
// "**A****B****C**", which is what real Claude Code transcripts showed.
func TestStreamingReasoningSummaryPartsAreSeparated(t *testing.T) {
	state := NewResponsesEventToAnthropicState()
	var all []AnthropicStreamEvent
	appendAll := func(events []AnthropicStreamEvent) { all = append(all, events...) }

	appendAll(ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:        "response.output_item.added",
		OutputIndex: 0,
		Item:        &ResponsesOutput{Type: "reasoning", ID: "rs_1"},
	}, state))

	for i, text := range []string{"**Investigating sessions**", "**Planning inspection**", "**Executing search**"} {
		appendAll(ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
			Type:         "response.reasoning_summary_part.added",
			OutputIndex:  0,
			SummaryIndex: i,
			Part:         &ResponsesContentPart{Type: "summary_text"},
		}, state))
		appendAll(ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
			Type:         "response.reasoning_summary_text.delta",
			OutputIndex:  0,
			SummaryIndex: i,
			Delta:        text,
		}, state))
		appendAll(ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
			Type:         "response.reasoning_summary_text.done",
			OutputIndex:  0,
			SummaryIndex: i,
		}, state))
	}

	got := collectThinking(all)
	assert.Equal(t, "**Investigating sessions**\n\n**Planning inspection**\n\n**Executing search**", got)
	assert.NotContains(t, got, "****", "adjacent summary parts must not run together")
}

// summary_index 0 opens the block, so it must never be prefixed with a blank line.
func TestStreamingReasoningFirstPartNotPrefixed(t *testing.T) {
	state := NewResponsesEventToAnthropicState()
	var all []AnthropicStreamEvent
	appendAll := func(events []AnthropicStreamEvent) { all = append(all, events...) }

	appendAll(ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:        "response.output_item.added",
		OutputIndex: 0,
		Item:        &ResponsesOutput{Type: "reasoning", ID: "rs_1"},
	}, state))
	appendAll(ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:         "response.reasoning_summary_part.added",
		OutputIndex:  0,
		SummaryIndex: 0,
		Part:         &ResponsesContentPart{Type: "summary_text"},
	}, state))
	appendAll(ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:         "response.reasoning_summary_text.delta",
		OutputIndex:  0,
		SummaryIndex: 0,
		Delta:        "only part",
	}, state))

	assert.Equal(t, "only part", collectThinking(all))
}

// A part event for an output index with no open thinking block must be dropped
// rather than emitting a delta against a wrong or unopened block index.
func TestStreamingReasoningPartWithoutOpenBlockIgnored(t *testing.T) {
	state := NewResponsesEventToAnthropicState()
	events := ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:         "response.reasoning_summary_part.added",
		OutputIndex:  7,
		SummaryIndex: 3,
		Part:         &ResponsesContentPart{Type: "summary_text"},
	}, state)
	assert.Empty(t, events)
}

// Three consecutive reasoning items each keep their own block and their own
// signature; separators must not leak across items. Real responses carry 2-3
// reasoning items before the function_call items.
func TestStreamingMultipleReasoningItemsKeepSeparateBlocks(t *testing.T) {
	state := NewResponsesEventToAnthropicState()
	var all []AnthropicStreamEvent
	appendAll := func(events []AnthropicStreamEvent) { all = append(all, events...) }

	for idx, spec := range []struct {
		id    string
		parts []string
		enc   string
	}{
		{"rs_1", []string{"first-a", "first-b"}, "enc-1"},
		{"rs_2", nil, "enc-2"}, // empty summary with a valid signature
		{"rs_3", []string{"third"}, "enc-3"},
	} {
		appendAll(ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
			Type:        "response.output_item.added",
			OutputIndex: idx,
			Item:        &ResponsesOutput{Type: "reasoning", ID: spec.id},
		}, state))
		for i, text := range spec.parts {
			appendAll(ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
				Type:         "response.reasoning_summary_part.added",
				OutputIndex:  idx,
				SummaryIndex: i,
				Part:         &ResponsesContentPart{Type: "summary_text"},
			}, state))
			appendAll(ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
				Type:         "response.reasoning_summary_text.delta",
				OutputIndex:  idx,
				SummaryIndex: i,
				Delta:        text,
			}, state))
		}
		appendAll(ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
			Type:        "response.output_item.done",
			OutputIndex: idx,
			Item: &ResponsesOutput{
				Type:             "reasoning",
				ID:               spec.id,
				EncryptedContent: spec.enc,
				Status:           "completed",
			},
		}, state))
	}

	blocks := map[int]string{}
	signatures := map[int]string{}
	for _, ev := range all {
		if ev.Type != "content_block_delta" || ev.Delta == nil || ev.Index == nil {
			continue
		}
		switch ev.Delta.Type {
		case "thinking_delta":
			blocks[*ev.Index] += ev.Delta.Thinking
		case "signature_delta":
			signatures[*ev.Index] = ev.Delta.Signature
		}
	}

	require.Len(t, signatures, 3, "each reasoning item keeps its own block and signature")
	assert.Equal(t, "first-a\n\nfirst-b", blocks[0])
	assert.Equal(t, "third", blocks[2])
	assert.Equal(t, "enc-1", signatures[0])
	assert.Equal(t, "enc-2", signatures[1])
	assert.Equal(t, "enc-3", signatures[2])
}

// The buffered converter must join parts exactly like the streaming one so a
// non-streaming client sees byte-identical thinking text.
func TestResponsesToAnthropicJoinsSummaryPartsWithBlankLine(t *testing.T) {
	resp := &ResponsesResponse{
		Output: []ResponsesOutput{{
			Type: "reasoning",
			ID:   "rs_1",
			Summary: []ResponsesSummary{
				{Type: "summary_text", Text: "**A**"},
				{Type: "summary_text", Text: "**B**"},
				{Type: "summary_text", Text: "**C**"},
			},
			EncryptedContent: "enc-1",
		}},
	}

	out := ResponsesToAnthropic(resp, "claude-opus-5")
	require.NotEmpty(t, out.Content)
	assert.Equal(t, "thinking", out.Content[0].Type)
	assert.Equal(t, "**A**\n\n**B**\n\n**C**", out.Content[0].Thinking)
	assert.Equal(t, "enc-1", out.Content[0].Signature)
}

// A reasoning item with no visible summary still yields a signature-only
// thinking block; the separator must not introduce stray whitespace.
func TestResponsesToAnthropicSignatureOnlyThinkingUnchanged(t *testing.T) {
	resp := &ResponsesResponse{
		Output: []ResponsesOutput{{
			Type:             "reasoning",
			ID:               "rs_1",
			EncryptedContent: "enc-only",
		}},
	}

	out := ResponsesToAnthropic(resp, "claude-opus-5")
	require.NotEmpty(t, out.Content)
	assert.Equal(t, "thinking", out.Content[0].Type)
	assert.Equal(t, "", out.Content[0].Thinking)
	assert.Equal(t, "enc-only", out.Content[0].Signature)
}
