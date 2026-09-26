import { motion } from 'framer-motion'
import { Sparkle, CheckCircle, Warning, Clock, Scales, CaretRight, ShieldCheck, Coin } from '@phosphor-icons/react'
import ReactMarkdown from 'react-markdown'
import styles from './PromptOptimizer.module.css'

export default function PromptOptimizer({ isOptimizing, error, rawPrompt, optimizationData, onSelectPrompt, onBack }) {
  if (isOptimizing) {
    return (
      <motion.div 
        className={styles.optimizingPanel}
        initial={{ opacity: 0, height: 0 }}
        animate={{ opacity: 1, height: 'auto' }}
        exit={{ opacity: 0, height: 0 }}
      >
        <div className={styles.scannerLine} />
        <div className={styles.sparkleWrap}>
          <Sparkle size={32} weight="duotone" className={styles.sparkleIcon} />
        </div>
        <h3>Prompt Optimization in Process</h3>
        <p>Analyzing cognitive structure and restructuring for maximum efficiency...</p>
        <div className={styles.gridContainer}>
          <div className={styles.gridCell} />
          <div className={styles.gridCell} />
          <div className={styles.gridCell} />
        </div>
      </motion.div>
    )
  }

  if (error) {
    return (
      <motion.div 
        className={styles.errorPanel}
        initial={{ opacity: 0, height: 0 }}
        animate={{ opacity: 1, height: 'auto' }}
        exit={{ opacity: 0, height: 0 }}
      >
        <Warning size={32} color="var(--error)" weight="duotone" />
        <h3>Optimization Failed</h3>
        <p>{error}</p>
        <button className={styles.backBtn} onClick={onBack}>Dismiss</button>
      </motion.div>
    )
  }

  if (!optimizationData) return null

  const { original, suggested } = optimizationData
  
  const tokenDiff = original.tokens_used - suggested.tokens_used
  const isCompressed = tokenDiff > 0
  const badgeText = isCompressed ? `-${tokenDiff.toFixed(1)} Removed` : `+${Math.abs(tokenDiff).toFixed(1)} Added`

  return (
    <motion.div 
      className={styles.inlinePanel}
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      exit={{ opacity: 0, height: 0 }}
      style={{ overflow: 'hidden' }}
    >
      <div className={styles.header}>
        <Sparkle size={24} weight="duotone" style={{ color: '#00e5ff' }} />
        <h2>Optimization Analytics</h2>
        <button className={styles.closeBtn} onClick={onBack}>✕</button>
      </div>

        <div className={styles.comparisonGrid}>
          {/* ORIGINAL */}
          <div className={`${styles.card} ${styles.cardOriginal}`}>
            <div className={styles.cardHeader}>
              <span className={styles.badgeLabel}>Original</span>
            </div>
            <div className={styles.promptView}>
              <ReactMarkdown>{original.prompt}</ReactMarkdown>
            </div>
            <div className={styles.metrics}>
              <div className={styles.metric}>
                <Scales size={16} /> <span>Weight: {original.tier_assigned}</span>
              </div>
              <div className={styles.metric}>
                <Coin size={16} /> <span>Tokens: {original.tokens_used}</span>
              </div>
            </div>
            <button 
              className={styles.selectBtn}
              onClick={() => onSelectPrompt(original.prompt)}
            >
              Send Original <CaretRight size={16} />
            </button>
          </div>

          {/* SUGGESTED */}
          <div className={`${styles.card} ${styles.cardSuggested}`}>
             <div className={styles.cardHeader}>
              <span className={styles.badgeLabelActive}>Optimized</span>
              <span 
                className={styles.savingBadge} 
                style={{
                  color: isCompressed ? '#00e676' : 'var(--text-dim)',
                  background: isCompressed ? 'rgba(0, 230, 118, 0.1)' : 'rgba(255, 255, 255, 0.05)'
                }}
              >
                 {badgeText}
              </span>
            </div>
            <div className={styles.promptView}>
              <ReactMarkdown>{suggested.prompt}</ReactMarkdown>
            </div>
             <div className={styles.metrics}>
              <div className={styles.metric}>
                <ShieldCheck size={16} style={{ color: '#00e676'}} /> 
                <span style={{ color: '#00e676'}}>Weight: {suggested.tier_assigned}</span>
              </div>
              <div className={styles.metric}>
                <Coin size={16} style={{ color: '#00e5ff'}} /> 
                <span style={{ color: '#00e5ff'}}>Tokens: {suggested.tokens_used}</span>
              </div>
            </div>
            <button 
              className={styles.selectBtnPrimary}
              onClick={() => onSelectPrompt(suggested.prompt)}
            >
              Use Optimized <Sparkle size={16} weight="fill" />
            </button>
          </div>
        </div>
    </motion.div>
  )
}
