import { motion } from 'framer-motion'
import { useEffect, useState, useRef, useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import {
  ArrowRight,
  Brain,
  Gauge,
  Sparkle,
  Target,
} from '@phosphor-icons/react'
import styles from './ResultsPanel.module.css'

function normalizeRoutingReason(reason = '') {
  return reason
    .replace(/[\u{1F300}-\u{1F6FF}\u{1F900}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, '')
    .replace(/â†'/g, '→')
    .replace(/â†'/g, '↑')
    .replace(/â†"/g, '↓')
    .replace(/Â·/g, '·')
    .replace(/\s+/g, ' ')
    .trim()
}

function parseMetric(value = '') {
  const match = value.match(/^(.+?)\s*\(([^)]+)\)$/)
  if (!match) return { label: value, score: '' }
  return { label: match[1].trim(), score: match[2].trim() }
}

const MODEL_MAP = {
  'Gemma 4 26B (a4b)': 'Edge Neural Unit',
  'Gemma 4 26B (OpenRouter)': 'Edge Fallback Unit',
  'Gemini 3.5 Flash Lite': 'Logical Flash Core',
  'Google Gemma 2 9B (Free)': 'Logical Fallback Core',
  'Llama 3.1 8B (OpenRouter)': 'Logical Fallback Core',
  'DeepSeek V4.1 Flash': 'Multimodal Flash Engine',
  'Nemotron Mini 4B': 'Edge Processing Unit',
  'Gemma 3n E4B': 'Edge Fallback Unit',
  'Nemotron Nano 9B v2': 'Logical Reasoning Core',
  'Nemotron Nano 30B-A3B': 'Core Analytical Unit',
  'Muse Glimmer 30B': 'Core Analytical Unit',
  'Nemotron 3.5 Lightning 30B': 'Analytical Fallback Core',
  'Nemotron 3 Super 120B': 'Deep Reasoning Engine',
  'GLM 5.3 Flash': 'Flash Reasoning Engine',
  'Mistral Medium 3.5': 'Reasoning Fallback Engine',
  'Qwen3 Coder 480B': 'Frontier Code Nexus',
  'Kimi K3': 'Frontier Multimodal Nexus',
  'Nemotron 3 Ultra 550B': 'Frontier Ultra Engine',
  'Qwen3.5 397B': 'Frontier Fallback Nexus'
}

function parseRoutingReason(reason = '') {
  const cleaned = normalizeRoutingReason(reason)
  const parts = cleaned.split(/\s*·\s*/).filter(Boolean)

  const data = {
    taskType: '',
    primarySignal: null,
    factors: [],
    confidence: '',
    scorer: '',
    adjustment: '',
    routeTier: '',
    routeModel: '',
    fallback: cleaned,
  }

  parts.forEach((part) => {
    if (part.startsWith('Detected task type:')) {
      data.taskType = part.replace('Detected task type:', '').trim()
      return
    }
    if (part.startsWith('Primary signal:')) {
      data.primarySignal = parseMetric(part.replace('Primary signal:', '').trim())
      return
    }
    if (part.startsWith('Contributing factors:')) {
      data.factors = part
        .replace('Contributing factors:', '')
        .split(',')
        .map((factor) => parseMetric(factor.trim()))
        .filter((factor) => factor.label)
      return
    }
    if (part.startsWith('Classification confidence:')) {
      data.confidence = part.replace('Classification confidence:', '').trim()
      return
    }
    if (part.startsWith('Scorer:')) {
      data.scorer = part.replace('Scorer:', '').trim()
      return
    }
    if (part.startsWith('Task-type adjustment:')) {
      data.adjustment = part.replace('Task-type adjustment:', '').trim()
      return
    }
    if (part.startsWith('Routed to')) {
      const route = part.replace('Routed to', '').trim()
      const [rawTier, rawModel] = route.split(/\s*→\s*/)
      data.routeTier = TIER_MAP[rawTier?.trim()] || rawTier?.trim() || route
      data.routeModel = MODEL_MAP[rawModel?.trim()] || rawModel?.trim() || ''
    }
  })

  return data
}

function confidenceTone(confidence = '') {
  const lower = confidence.toLowerCase()
  if (lower.includes('high')) return 'high'
  if (lower.includes('medium')) return 'medium'
  if (lower.includes('low')) return 'low'
  return 'neutral'
}

function ReasoningInsight({ routingReason }) {
  const insight = parseRoutingReason(routingReason)
  const tone = confidenceTone(insight.confidence)

  if (!insight.taskType && !insight.primarySignal && !insight.routeTier) {
    return (
      <div className={styles.reasoningBlock}>
        <div className={styles.reasoningHeader}>Reasoning</div>
        <div className={styles.reasoningText}>{insight.fallback}</div>
      </div>
    )
  }

  return (
    <div className={styles.reasoningBlock}>
      <div className={styles.reasoningTop}>
        <div className={styles.reasoningTitleWrap}>
          <span className={styles.reasoningIcon}>
            <Brain size={16} weight="duotone" />
          </span>
          <div>
            <div className={styles.reasoningHeader}>Reasoning</div>
            <div className={styles.reasoningSubhead}>{insight.taskType || 'Task analysis'}</div>
          </div>
        </div>
        {insight.confidence && (
          <span className={`${styles.confidencePill} ${styles[`confidence${tone}`]}`}>
            {insight.confidence}
          </span>
        )}
      </div>

      {(insight.routeTier || insight.routeModel) && (
        <div className={styles.routeStrip}>
          <span>{insight.routeTier}</span>
          {insight.routeModel && (
            <>
              <ArrowRight size={14} weight="bold" />
              <strong>{insight.routeModel}</strong>
            </>
          )}
        </div>
      )}

      <div className={styles.reasoningMetrics}>
        {insight.primarySignal && (
          <div className={styles.reasoningMetric}>
            <span className={styles.metricIcon}><Target size={15} weight="duotone" /></span>
            <div>
              <span>Primary signal</span>
              <strong>{insight.primarySignal.label}</strong>
              {insight.primarySignal.score && <small>{insight.primarySignal.score}</small>}
            </div>
          </div>
        )}
        {insight.adjustment && (
          <div className={styles.reasoningMetric}>
            <span className={styles.metricIcon}><Sparkle size={15} weight="duotone" /></span>
            <div>
              <span>Adjustment</span>
              <strong>{insight.adjustment}</strong>
            </div>
          </div>
        )}
      </div>

      {insight.factors.length > 0 && (
        <div className={styles.factorGroup}>
          <div className={styles.factorLabel}>Contributing factors</div>
          <div className={styles.factorChips}>
            {insight.factors.map((factor) => (
              <span key={`${factor.label}-${factor.score}`} className={styles.factorChip}>
                {factor.label}
                {factor.score && <b>{factor.score}</b>}
              </span>
            ))}
          </div>
        </div>
      )}

      {insight.scorer && (
        <div className={styles.reasoningFoot}>
          <Gauge size={14} weight="duotone" />
          <span>Scorer</span>
          <strong>{insight.scorer}</strong>
        </div>
      )}
    </div>
  )
}

const TIER_MAP = {
  'Edge Node': 'Tier 0',
  'Logical Core': 'Tier 1',
  'Analytical Engine': 'Tier 2',
  'Complex Reasoning': 'Tier 3',
  'Max Cognition': 'Tier 4',
  'Tier 0': 'Tier 0',
  'Tier 1': 'Tier 1',
  'Tier 2': 'Tier 2',
  'Tier 3': 'Tier 3',
  'Tier 4': 'Tier 4'
}

// ── Markdown Response Component ─────────────────────────────────────────────
// Renders LLM output with full markdown: tables, code blocks, headers, lists, etc.
function MarkdownResponse({ text, isStreaming }) {
  const containerRef = useRef(null)
  const [displayText, setDisplayText] = useState('')
  const [isRevealing, setIsRevealing] = useState(true)
  const revealIntervalRef = useRef(null)
  const lastTextRef = useRef('')

  useEffect(() => {
    if (!text) {
      setDisplayText('')
      setIsRevealing(true)
      return
    }

    // If text changed (new response or streaming chunk)
    if (text !== lastTextRef.current) {
      const prevLen = lastTextRef.current.length
      lastTextRef.current = text

      // If entirely new text (not a streaming append), reveal progressively
      if (prevLen === 0 && text.length > 0) {
        setIsRevealing(true)
        let charIdx = 0
        const totalChars = text.length
        // Reveal ~80 chars per tick at 16ms = ~5000 chars/sec
        const charsPerTick = Math.max(20, Math.ceil(totalChars / 60))

        if (revealIntervalRef.current) cancelAnimationFrame(revealIntervalRef.current)

        const reveal = () => {
          charIdx = Math.min(charIdx + charsPerTick, totalChars)
          setDisplayText(text.slice(0, charIdx))
          if (charIdx < totalChars) {
            revealIntervalRef.current = requestAnimationFrame(reveal)
          } else {
            setIsRevealing(false)
          }
        }
        revealIntervalRef.current = requestAnimationFrame(reveal)
      } else {
        // Streaming append — just show the latest text
        setDisplayText(text)
      }
    }

    return () => {
      if (revealIntervalRef.current) cancelAnimationFrame(revealIntervalRef.current)
    }
  }, [text])

  // Auto-scroll during streaming/revealing
  useEffect(() => {
    if (containerRef.current && (isStreaming || isRevealing)) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [displayText, isStreaming, isRevealing])

  if (!text) {
    return (
      <div className={styles.responseBody} ref={containerRef}>
        <span className={styles.typingIndicator}>
          <span /><span /><span />
        </span>
      </div>
    )
  }

  return (
    <div className={styles.responseBody} ref={containerRef}>
      <div className={styles.markdownContent}>
        <ReactMarkdown 
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeKatex]}
        >
          {displayText}
        </ReactMarkdown>
      </div>
      {isStreaming && (
        <span className={styles.cursorBlink}>▊</span>
      )}
    </div>
  )
}

export default function ResultsPanel({ query, result, onClear }) {
  if (!result) return null

  const {
    response,
    model_used,
    tier_assigned,
    budget_score,
    tokens_saved,
    latency_ms,
    cognitive_profile,
    routing_reason,
  } = result

  const displayTier = TIER_MAP[tier_assigned] || tier_assigned
  const isStreaming = model_used === 'Routing...' || (response && !latency_ms)

  return (
    <motion.section
      className={styles.section}
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
    >
      <div className={styles.header}>
        <h2 className={styles.title}>Analysis &amp; Response</h2>
        <button className={styles.clearBtn} onClick={onClear}>Clear</button>
      </div>

      <div className={styles.grid}>
        {/* Left Column (now 1fr): Query & Response */}
        <div className={styles.responseCol}>
          {query && (
            <div className={styles.queryCard}>
              <div className={styles.cardHeader}>
                <div className={styles.cardTitle}>Original Query</div>
              </div>
              <div className={styles.queryBody}>
                {query}
              </div>
            </div>
          )}

          <div className={styles.responseCard}>
            <div className={styles.cardHeader}>
              <div className={styles.cardTitle}>Generated Output</div>
              <div className={styles.badge}>{model_used}</div>
            </div>
            <MarkdownResponse text={response} isStreaming={isStreaming} />
          </div>
        </div>

        {/* Right Column (now 380px): Telemetry & Insight */}
        <div className={styles.telemetryCol}>
          
          {/* Routing Decision */}
          <div className={styles.metaCard}>
            <div className={styles.cardTitle}>Routing Decision</div>
            <div className={styles.statList}>
              <div className={styles.statListItem}>
                <span className={styles.statListLabel}>Tier Assigned</span>
                <span className={styles.statListValue}>{displayTier}</span>
              </div>
              <div className={styles.statListItem}>
                <span className={styles.statListLabel}>Budget Score</span>
                <span className={styles.statListValue}>{budget_score}/100</span>
              </div>
              <div className={styles.statListItem}>
                <span className={styles.statListLabel}>Latency</span>
                <span className={styles.statListValue}>{latency_ms}ms</span>
              </div>
              <div className={styles.statListItem}>
                <span className={styles.statListLabel}>Tokens Saved</span>
                <span className={styles.statListValue} style={{ color: 'var(--success)' }}>+{tokens_saved}</span>
              </div>
            </div>
            
            <ReasoningInsight routingReason={routing_reason} />
          </div>

          {/* Cognitive Profile */}
          <div className={styles.metaCard}>
            <div className={styles.cardTitle}>Cognitive Profile</div>
            <div className={styles.barsGrid}>
              {Object.entries(cognitive_profile || {})
                .filter(([k]) => !['task_type', 'signals', 'confidence', 'scorer_used', 'complexity_breakdown', 'cora_score'].includes(k))
                .map(([key, val]) => (
                <div key={key} className={styles.barWrap}>
                  <div className={styles.barLabel}>
                    {key.replace(/_/g, ' ')} <span>{val}/100</span>
                  </div>
                  <div className={styles.track}>
                    <motion.div
                      className={styles.fill}
                      initial={{ width: 0 }}
                      animate={{ width: `${val}%` }}
                      transition={{ duration: 1, delay: 0.2 }}
                    />
                  </div>
                </div>
              ))}
            </div>
            {/* Signals */}
            {(cognitive_profile || {}).signals && (cognitive_profile || {}).signals.length > 0 && (
              <div className={styles.signals}>
                {(cognitive_profile || {}).signals.map((sig, i) => (
                  <span key={i} className={styles.signalBadge}>{sig}</span>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </motion.section>
  )
}
