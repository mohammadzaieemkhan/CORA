import { useState, useEffect, forwardRef, useImperativeHandle } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { fetchHistory, deleteHistoryItem } from '../services/api'
import styles from './Sidebar.module.css'

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

const MODEL_MAP = {
  'Nemotron Mini 4B': 'Edge',
  'Gemma 3n E4B': 'Edge',
  'Nemotron Nano 9B v2': 'Logical',
  'Nemotron Nano 30B-A3B': 'Analytical',
  'Nemotron 3 Super 120B': 'Reasoning',
  'Mistral Medium 3.5': 'Reasoning',
  'Qwen3 Coder 480B': 'Deep',
  'Qwen3.5 397B': 'Deep'
}

const Sidebar = forwardRef(({ onSelect }, ref) => {
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [deletingId, setDeletingId] = useState(null)

  const load = async () => {
    try {
      const token = localStorage.getItem('cora_token')
      if (!token) {
        setHistory([])
        setLoading(false)
        return
      }
      const data = await fetchHistory()
      setHistory(data.queries || [])
    } catch (err) {
      console.error("Failed to load history", err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  // Expose addToHistory so parent can push new queries in real-time
  useImperativeHandle(ref, () => ({
    addToHistory(item) {
      setHistory(prev => [item, ...prev])
    },
    reload() {
      load()
    }
  }))

  const handleDelete = async (e, id) => {
    e.stopPropagation()
    setDeletingId(id)
    try {
      await deleteHistoryItem(id)
      setHistory(prev => prev.filter(item => item.id !== id))
    } catch (err) {
      console.error("Failed to delete", err)
    }
    setDeletingId(null)
  }

  return (
    <aside className={styles.sidebar}>
      <div className={styles.header}>
        <div className={styles.title}>History</div>
        <span className={styles.count}>{history.length}</span>
      </div>
      {loading ? (
        <div className={styles.empty}>
          <div className={styles.loadingDots}>
            <span /><span /><span />
          </div>
        </div>
      ) : history.length === 0 ? (
        <div className={styles.empty}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="32" height="32" style={{opacity:0.3, marginBottom: 8}}>
            <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          <span>No conversations yet</span>
        </div>
      ) : (
        <div className={styles.list}>
          <AnimatePresence>
            {history.map((item) => {
              const label = item.tier_assigned || MODEL_MAP[item.model_used] || item.model_used || 'Unknown';
              return (
                <motion.div
                  key={item.id}
                  className={styles.item}
                  onClick={() => onSelect(item)}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20, height: 0, marginBottom: 0, padding: 0 }}
                  transition={{ duration: 0.25 }}
                  layout
                >
                  <div className={styles.prompt}>{item.prompt}</div>
                  <div className={styles.meta}>
                    <span className={styles.badge}>{label}</span>
                    <div className={styles.metaRight}>
                      <span className={styles.date}>
                        {new Date(item.created_at).toLocaleDateString()}
                      </span>
                      <button
                        className={styles.deleteBtn}
                        onClick={(e) => handleDelete(e, item.id)}
                        disabled={deletingId === item.id}
                        title="Delete"
                      >
                        {deletingId === item.id ? (
                          <span className={styles.miniSpinner} />
                        ) : (
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="12" height="12">
                            <path d="M3 6h18m-2 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
                          </svg>
                        )}
                      </button>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>
      )}
    </aside>
  )
})

Sidebar.displayName = 'Sidebar'
export default Sidebar
