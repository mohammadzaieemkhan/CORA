import { motion, AnimatePresence } from 'framer-motion'
import { useEffect, useState, useRef, useCallback } from 'react'
import { fetchStats, fetchHistory } from '../services/api'
import {
  ChartBar,
  Lightning,
  Crosshair,
  Timer,
  CurrencyDollar,
  Robot,
  Trophy,
  Star,
  SquaresFour,
  ArrowUp,
  ArrowDown,
  Minus,
  Pulse,
} from '@phosphor-icons/react'
import styles from './Dashboard.module.css'

// ── SVG Radar Chart (6 dimensions) ──────────────────────────────────────────
function RadarChart({ profile, size = 200 }) {
  const dims = [
    { key: 'reasoning_depth', label: 'Reasoning' },
    { key: 'domain_specificity', label: 'Domain' },
    { key: 'code_complexity', label: 'Code' },
    { key: 'creative_demand', label: 'Creative' },
    { key: 'precision_required', label: 'Precision' },
    { key: 'structural_complexity', label: 'Structure' },
  ]
  const cx = size / 2, cy = size / 2, r = size * 0.38
  const angleStep = (Math.PI * 2) / dims.length

  const getPoint = (val, i) => {
    const angle = angleStep * i - Math.PI / 2
    const dist = (val / 100) * r
    return { x: cx + Math.cos(angle) * dist, y: cy + Math.sin(angle) * dist }
  }

  const gridLevels = [20, 40, 60, 80, 100]
  const values = dims.map(d => profile[d.key] || 0)
  const points = values.map((v, i) => getPoint(v, i))
  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ') + 'Z'

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className={styles.radarSvg}>
      {/* Grid */}
      {gridLevels.map(level => {
        const pts = dims.map((_, i) => getPoint(level, i))
        const d = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ') + 'Z'
        return <path key={level} d={d} fill="none" stroke="var(--border)" strokeWidth="0.5" opacity={0.5} />
      })}
      {/* Axes */}
      {dims.map((_, i) => {
        const end = getPoint(100, i)
        return <line key={i} x1={cx} y1={cy} x2={end.x} y2={end.y} stroke="var(--border)" strokeWidth="0.5" opacity={0.3} />
      })}
      {/* Data polygon */}
      <motion.path
        d={pathD}
        fill="var(--accent)"
        fillOpacity={0.15}
        stroke="var(--accent)"
        strokeWidth="2"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: 1, opacity: 1 }}
        transition={{ duration: 1.2, ease: 'easeOut' }}
      />
      {/* Data dots */}
      {points.map((p, i) => (
        <motion.circle
          key={i}
          cx={p.x} cy={p.y} r={3}
          fill="var(--accent)"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2 + i * 0.1 }}
        />
      ))}
      {/* Labels */}
      {dims.map((dim, i) => {
        const labelPt = getPoint(120, i)
        return (
          <text
            key={dim.key}
            x={labelPt.x} y={labelPt.y}
            textAnchor="middle"
            dominantBaseline="middle"
            className={styles.radarLabel}
          >
            {dim.label} ({profile[dim.key] || 0})
          </text>
        )
      })}
    </svg>
  )
}

// ── Donut Chart ─────────────────────────────────────────────────────────────
function DonutChart({ data, size = 180, strokeWidth = 22 }) {
  const r = (size - strokeWidth) / 2
  const circumference = 2 * Math.PI * r
  const cx = size / 2, cy = size / 2
  const total = Object.values(data).reduce((a, b) => a + b, 0) || 1

  const colors = [
    '#00e5ff',       // T0 — cyan
    '#00e676',       // T1 — green
    '#ffab00',       // T2 — amber
    '#7c4dff',       // T3 — purple
    '#ff006e',       // T4 — pink
  ]

  let offset = 0
  const segments = Object.entries(data).map(([tier, count], i) => {
    const pct = count / total
    const dashLength = pct * circumference
    const seg = { tier, count, pct, dashLength, offset, color: colors[i % colors.length] }
    offset += dashLength
    return seg
  })

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {/* Background ring — always visible in both themes */}
      <circle
        cx={cx} cy={cy} r={r}
        fill="none"
        stroke="#e0e0e0"
        strokeWidth={strokeWidth}
        opacity={0.25}
      />
      {segments.map((seg, i) => (
        <motion.circle
          key={seg.tier}
          cx={cx} cy={cy} r={r}
          fill="none"
          stroke={seg.color}
          strokeWidth={strokeWidth}
          strokeDasharray={`${seg.dashLength} ${circumference - seg.dashLength}`}
          strokeDashoffset={-seg.offset}
          strokeLinecap="round"
          transform={`rotate(-90 ${cx} ${cy})`}
          initial={{ strokeDasharray: `0 ${circumference}` }}
          animate={{ strokeDasharray: `${seg.dashLength} ${circumference - seg.dashLength}` }}
          transition={{ duration: 1, delay: i * 0.15, ease: 'easeOut' }}
        />
      ))}
      <text x={cx} y={cy - 6} textAnchor="middle" className={styles.donutCenter}>{total}</text>
      <text x={cx} y={cy + 12} textAnchor="middle" className={styles.donutLabel}>total</text>
    </svg>
  )
}

// ── Mini Sparkline ──────────────────────────────────────────────────────────
function Sparkline({ data, width = 120, height = 32, color = 'var(--accent)' }) {
  if (!data || data.length < 2) return null
  const max = Math.max(...data, 1)
  const min = Math.min(...data, 0)
  const range = max - min || 1
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width
    const y = height - ((v - min) / range) * (height - 4) - 2
    return `${x},${y}`
  }).join(' ')

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className={styles.sparkline}>
      <defs>
        <linearGradient id="sparkGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <polygon
        points={`0,${height} ${points} ${width},${height}`}
        fill="url(#sparkGrad)"
      />
    </svg>
  )
}

// ── Animated Number ─────────────────────────────────────────────────────────
function AnimatedNumber({ value, suffix = '', decimals = 0 }) {
  const [display, setDisplay] = useState(0)
  const ref = useRef(null)

  useEffect(() => {
    const target = typeof value === 'number' ? value : parseFloat(value) || 0
    const start = display
    const duration = 800
    const startTime = performance.now()

    const animate = (now) => {
      const elapsed = now - startTime
      const progress = Math.min(elapsed / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3) // ease-out cubic
      setDisplay(start + (target - start) * eased)
      if (progress < 1) ref.current = requestAnimationFrame(animate)
    }
    ref.current = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(ref.current)
  }, [value])

  return <>{decimals > 0 ? display.toFixed(decimals) : Math.round(display)}{suffix}</>
}

// ── Tier Bar ────────────────────────────────────────────────────────────────
function TierBar({ tier, count, total, color, delay }) {
  const pct = total > 0 ? (count / total) * 100 : 0
  return (
    <div className={styles.tierRow}>
      <span className={styles.tierName}>{tier}</span>
      <div className={styles.tierTrack}>
        <motion.div
          className={styles.tierFill}
          style={{ background: color }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, delay, ease: [0.16, 1, 0.3, 1] }}
        />
      </div>
      <span className={styles.tierCount}>{count}</span>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN DASHBOARD
// ═══════════════════════════════════════════════════════════════════════════════
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

export default function Dashboard() {
  const [stats, setStats] = useState({
    total_queries: 0,
    total_tokens_saved: 0,
    average_budget_score: 0,
    routing_distribution: { 'Tier 0': 0, 'Tier 1': 0, 'Tier 2': 0, 'Tier 3': 0, 'Tier 4': 0 },
  })
  const [history, setHistory] = useState([])
  const [avgProfile, setAvgProfile] = useState(null)
  const [scoreHistory, setScoreHistory] = useState([])
  const [lastUpdated, setLastUpdated] = useState(null)
  const [prevStats, setPrevStats] = useState(null)

  const computeInsights = useCallback((queries) => {
    if (!queries.length) return

    // Average cognitive profile across all queries
    const profileKeys = ['reasoning_depth', 'domain_specificity', 'code_complexity', 'creative_demand', 'precision_required', 'structural_complexity']
    const avg = {}
    profileKeys.forEach(k => {
      const vals = queries.filter(q => q.cognitive_profile && q.cognitive_profile[k] != null).map(q => q.cognitive_profile[k])
      avg[k] = vals.length > 0 ? Math.round(vals.reduce((a, b) => a + b, 0) / vals.length) : 0
    })
    setAvgProfile(avg)

    const recent = queries.slice(0, 20).reverse()
    setScoreHistory(recent.map(q => q.budget_score || 0))
  }, [])

  useEffect(() => {
    const loadAll = async () => {
      try {
        const token = localStorage.getItem('cora_token')

        // Stats are public — always fetch
        const s = await fetchStats()

        // Map raw tiers to descriptive names
        if (s.routing_distribution) {
          const mappedDist = {}
          Object.entries(s.routing_distribution).forEach(([k, v]) => {
            mappedDist[TIER_MAP[k] || k] = v
          })
          s.routing_distribution = mappedDist
        }

        // Track delta for live indicator
        setPrevStats(prev => {
          if (prev && prev.total_queries !== s.total_queries) {
            setLastUpdated(new Date())
          }
          return stats
        })
        setStats(s)

        // History requires auth — only fetch if logged in
        if (token) {
          const h = await fetchHistory(1, 50)
          // Guard against 401 response (returns {detail: ...} instead of {queries: ...})
          const queries = h.queries || []
          setHistory(queries)
          computeInsights(queries)
        }
      } catch (e) {
        console.error('Dashboard load error', e)
      }
    }
    loadAll()
    const id = setInterval(loadAll, 12000) // Poll every 12s, not 8s
    return () => clearInterval(id)
  }, [computeInsights])

  const dist = stats.routing_distribution || {}
  const totalDist = Object.values(dist).reduce((a, b) => a + b, 0) || 1

  const tierColors = [
    '#00e5ff',       // T0 — cyan
    '#00e676',       // T1 — green
    '#ffab00',       // T2 — amber
    '#7c4dff',       // T3 — purple
    '#ff006e',       // T4 — pink
  ]

  // Compute additional metrics from live data
  const totalTokensUsed = history.reduce((a, q) => a + (q.tokens_used || 0), 0)
  const totalTokensSavedLocal = history.reduce((a, q) => a + (q.tokens_saved || 0), 0)
  const savingsRate = (totalTokensUsed + totalTokensSavedLocal) > 0
    ? ((totalTokensSavedLocal / (totalTokensUsed + totalTokensSavedLocal)) * 100).toFixed(1)
    : 0
  const taskTypes = {}
  history.forEach(q => {
    if (q.task_type) taskTypes[q.task_type] = (taskTypes[q.task_type] || 0) + 1
  })
  const topTask = Object.entries(taskTypes).sort((a, b) => b[1] - a[1])[0]

  // Find most used model
  const modelCounts = {}
  history.forEach(q => {
    if (q.model_used) modelCounts[q.model_used] = (modelCounts[q.model_used] || 0) + 1
  })

  const MODEL_MAP = {
    'Nemotron Mini 4B': 'Edge Processing Unit',
    'Gemma 3n E4B': 'Edge Fallback Unit',
    'Nemotron Nano 9B v2': 'Logical Reasoning Core',
    'Nemotron Nano 30B-A3B': 'Core Analytical Unit',
    'Nemotron 3 Super 120B': 'Deep Reasoning Engine',
    'Mistral Medium 3.5': 'Reasoning Fallback Engine',
    'Qwen3 Coder 480B': 'Frontier Code Nexus',
    'Qwen3.5 397B': 'Frontier Fallback Nexus'
  }

  const topModelRaw = Object.entries(modelCounts).sort((a, b) => b[1] - a[1])[0]
  const topModelName = topModelRaw ? (MODEL_MAP[topModelRaw[0]] || topModelRaw[0]) : null

  // Delta indicators
  const queryDelta = prevStats ? stats.total_queries - prevStats.total_queries : 0

  const hasData = stats.total_queries > 0

  return (
    <section className={styles.dashboard}>
      <div className={styles.sectionHeader}>
        <div className={styles.sectionTitle}>
          <SquaresFour size={18} weight="duotone" />
          <span>Analytics Dashboard</span>
        </div>
        <div className={styles.liveIndicator}>
          <Pulse size={14} weight="fill" />
          <span>Live</span>
          {lastUpdated && (
            <span className={styles.lastSync}>
              · synced {lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          )}
        </div>
      </div>

      {/* ── Row 1: Key Metrics ────────────────────── */}
      <div className={styles.metricsRow}>
        {[
          { label: 'Total Queries', value: stats.total_queries, Icon: ChartBar, color: 'var(--accent)', sparkData: scoreHistory, delta: queryDelta },
          { label: 'Tokens Saved', value: stats.total_tokens_saved, Icon: Lightning, color: 'var(--success)', sparkData: null },
          { label: 'Avg Budget Score', value: stats.average_budget_score, Icon: Crosshair, color: 'var(--warning)', decimals: 1, sparkData: scoreHistory },
          { label: 'Savings Rate', value: savingsRate, Icon: CurrencyDollar, suffix: '%', color: 'var(--accent-tertiary)', decimals: 1, sparkData: null },
          { label: 'Models Used', value: Object.keys(modelCounts).length, Icon: Robot, color: 'var(--info)', sparkData: null },
        ].map((m, i) => (
          <motion.div
            key={m.label}
            className={styles.metricCard}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.08, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            whileHover={{ y: -3, transition: { duration: 0.2 } }}
          >
            <div className={styles.metricTop}>
              <div className={styles.metricIcon} style={{ color: m.color }}>
                <m.Icon size={20} weight="duotone" />
              </div>
              {m.delta > 0 && (
                <span className={styles.deltaBadge}>
                  <ArrowUp size={10} weight="bold" />
                  +{m.delta}
                </span>
              )}
            </div>
            <div className={styles.metricValue} style={{ color: m.color }}>
              <AnimatedNumber value={m.value} suffix={m.suffix || ''} decimals={m.decimals || 0} />
            </div>
            <div className={styles.metricLabel}>{m.label}</div>
            {m.sparkData && m.sparkData.length > 1 && (
              <div className={styles.metricSpark}>
                <Sparkline data={m.sparkData} color={m.color} width={100} height={24} />
              </div>
            )}
            <div className={styles.metricGlow} style={{ background: m.color }} />
          </motion.div>
        ))}
      </div>

      {/* ── Row 2: Visualizations ──────────────────── */}
      <div className={styles.vizRow}>
        {/* Tier Distribution Donut */}
        <motion.div
          className={styles.vizCard}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3, duration: 0.5 }}
        >
          <div className={styles.vizTitle}>Tier Distribution</div>
          {hasData ? (
            <>
              <div className={styles.donutWrap}>
                <DonutChart data={dist} />
              </div>
              <div className={styles.legendGrid}>
                {Object.entries(dist).map(([tier, count], i) => (
                  <div key={tier} className={styles.legendItem}>
                    <span className={styles.legendDot} style={{ background: tierColors[i] }} />
                    <span className={styles.legendTier}>{tier}</span>
                    <span className={styles.legendVal}>{count}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className={styles.vizEmpty}>Submit a query to see distribution</div>
          )}
        </motion.div>

        {/* Cognitive Radar */}
        <motion.div
          className={styles.vizCard}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4, duration: 0.5 }}
        >
          <div className={styles.vizTitle}>Avg Cognitive Profile</div>
          {avgProfile ? (
            <div className={styles.radarWrap}>
              <RadarChart profile={avgProfile} size={200} />
            </div>
          ) : (
            <div className={styles.vizEmpty}>No cognitive data yet</div>
          )}
        </motion.div>

        {/* Tier Breakdown Bars */}
        <motion.div
          className={styles.vizCard}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5, duration: 0.5 }}
        >
          <div className={styles.vizTitle}>Routing Breakdown</div>
          {hasData ? (
            <>
              <div className={styles.tierBars}>
                {Object.entries(dist).map(([tier, count], i) => (
                  <TierBar
                    key={tier}
                    tier={tier}
                    count={count}
                    total={totalDist}
                    color={tierColors[i]}
                    delay={0.5 + i * 0.1}
                  />
                ))}
              </div>
              <div className={styles.insightRow}>
                {topTask && (
                  <div className={styles.insightChip}>
                    <Trophy size={16} weight="fill" style={{ color: '#ffab00' }} />
                    <span>Top Task</span>
                    <strong>{topTask[0]}</strong>
                  </div>
                )}
                {topModelName && (
                  <div className={styles.insightChip}>
                    <Star size={16} weight="fill" style={{ color: '#00e5ff' }} />
                    <span>Top Model</span>
                    <strong>{topModelName}</strong>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className={styles.vizEmpty}>Start querying to see routing data</div>
          )}
        </motion.div>
      </div>
    </section>
  )
}
