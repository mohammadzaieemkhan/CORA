import { motion } from 'framer-motion'
import {
  ArrowRight,
  Brain,
  Gauge,
  Sparkle,
  Target,
} from '@phosphor-icons/react'
import styles from './ResultsPanel.module.css'

/**
 * Convert markdown-formatted LLM output to clean rendered HTML.
 * Handles: **bold**, *italic*, `code`, ### headers, - lists, numbered lists.
 */
function formatResponse(text) {
  if (!text) return ''

  return text
    // Convert ### headers to styled lines
    .replace(/^#{1,3}\s+(.+)$/gm, '<strong class="resp-heading">$1</strong>')
    // Convert **bold** to <strong>
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // Convert *italic* to <em>  (but not inside already-handled **)
    .replace(/(?<!\*)\*([^*]+?)\*(?!\*)/g, '<em>$1</em>')
    // Convert `inline code` to <code>
    .replace(/`([^`]+)`/g, '<code class="resp-code">$1</code>')
    // Convert bullet lists (- item)
    .replace(/^[\-\*]\s+(.+)$/gm, '<span class="resp-bullet">• $1</span>')
    // Convert numbered lists (1. item)
    .replace(/^\d+\.\s+(.+)$/gm, '<span class="resp-bullet">$1</span>')
}

function normalizeRoutingReason(reason = '') {
  return reason
    .replace(/[\u{1F300}-\u{1F6FF}\u{1F900}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, '')
    .replace(/â†’/g, '→')
    .replace(/â†‘/g, '↑')
    .replace(/â†“/g, '↓')
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
  'Mistral 7B v3': 'Core Analytical Unit',
  'Mixtral 8x7B': 'Parallel Logic Core',
  'Devstral 123B (NVIDIA)': 'Deep Reasoning Nexus',
  'Gemma 2B': 'Edge Processing Unit',
  'Llama 3.1 8B': 'General Purpose Engine'
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
      data.routeTier = rawTier?.trim() || route
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

  const displayTier = tier_assigned
  const formattedHTML = formatResponse(response)

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
            <div
              className={styles.responseBody}
              dangerouslySetInnerHTML={{
                __html: formattedHTML
                  .split('\n')
                  .map(line => line || '<br/>')
                  .join('<br/>')
              }}
            />
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
              {Object.entries(cognitive_profile)
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
            {cognitive_profile.signals && cognitive_profile.signals.length > 0 && (
              <div className={styles.signals}>
                {cognitive_profile.signals.map((sig, i) => (
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
