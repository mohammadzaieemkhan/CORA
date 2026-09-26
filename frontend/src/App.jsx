import { useState, useEffect, useRef } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { ThemeProvider } from './context/ThemeContext'
import ParticleCanvas from './components/ParticleCanvas'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import Dashboard from './components/Dashboard'
import PipelineViz from './components/PipelineViz'
import BentoGrid from './components/BentoGrid'
import PromptInput from './components/PromptInput'
import ResultsPanel from './components/ResultsPanel'
import AuthModal from './components/AuthModal'
import ProfileModal from './components/ProfileModal'
import Sidebar from './components/Sidebar'
import Footer from './components/Footer'
import { submitQuery, logout } from './services/api'
import styles from './App.module.css'

function App() {
  const [user, setUser] = useState(null)
  const [authModalOpen, setAuthModalOpen] = useState(false)
  const [profileModalOpen, setProfileModalOpen] = useState(false)
  const [authMode, setAuthMode] = useState('login')
  const [isProcessing, setIsProcessing] = useState(false)
  const [result, setResult] = useState(null)
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [showDashboard, setShowDashboard] = useState(false)
  const sidebarRef = useRef(null)
  const contentRef = useRef(null)
  
  useEffect(() => {
    const saved = localStorage.getItem('cora_user')
    if (saved) setUser(saved)
  }, [])

  const handleLogout = async () => {
    await logout()
    localStorage.removeItem('cora_token')
    localStorage.removeItem('cora_user')
    setUser(null)
    setResult(null)
  }

  const handleQuery = async (promptText) => {
    setIsProcessing(true)
    setResult({
      response: '',
      model_used: 'Routing...',
      tier_assigned: 'Analyzing...',
      budget_score: 0,
      tokens_saved: 0,
      latency_ms: 0,
      cognitive_profile: {},
      routing_reason: 'Evaluating cognitive load...',
      prompt: promptText,
    })
    
    let currentResponse = ''
    
    try {
      const { streamQuery } = await import('./services/api')
      await streamQuery(
        promptText, 
        null,
        (meta) => {
          setResult(prev => ({ ...prev, ...meta }))
        },
        (token) => {
          currentResponse += token
          setResult(prev => ({ ...prev, response: currentResponse }))
        },
        (doneData) => {
          setResult(prev => {
            const finalResult = { ...prev, ...doneData, response: currentResponse }
            // Reload sidebar from database to get the real persisted entry
            if (sidebarRef.current) {
              sidebarRef.current.reload()
            }
            return finalResult
          })
          // Scroll to top of content to see the result
          if (contentRef.current) {
            contentRef.current.scrollTo({ top: 0, behavior: 'smooth' })
          }
        },
        (err) => {
          console.error(err)
          alert('Error processing query: ' + err)
        }
      )
    } catch (err) {
      console.error(err)
      alert('Error processing query: ' + err.message)
    }
    setIsProcessing(false)
  }

  const handleHistorySelect = (historyResult) => {
    if (historyResult) {
      setResult(typeof historyResult === 'string' ? JSON.parse(historyResult) : historyResult)
      setShowDashboard(false)
      if (contentRef.current) {
        contentRef.current.scrollTo({ top: 0, behavior: 'smooth' })
      }
    }
  }

  return (
    <ThemeProvider>
      <div className="aurora-bg">
        <div className="aurora-blob blob-1"></div>
        <div className="aurora-blob blob-2"></div>
        <div className="aurora-blob blob-3"></div>
      </div>
      <ParticleCanvas />
      
      <div className={styles.container}>
        <Navbar 
          user={user} 
          onLogin={() => { setAuthMode('login'); setAuthModalOpen(true); }}
          onRegister={() => { setAuthMode('register'); setAuthModalOpen(true); }}
          onProfileClick={() => setProfileModalOpen(true)}
          onLogout={handleLogout} 
          isSidebarOpen={isSidebarOpen}
          onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
          showDashboard={showDashboard}
          onToggleDashboard={() => setShowDashboard(!showDashboard)}
        />
        
        {user ? (
          <div className={styles.authLayout}>
            <AnimatePresence>
              {isSidebarOpen && (
                <motion.div 
                  initial={{ width: 0, opacity: 0 }}
                  animate={{ width: 300, opacity: 1 }}
                  exit={{ width: 0, opacity: 0 }}
                  transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                  style={{ 
                    overflow: 'hidden',
                    flexShrink: 0,
                    position: 'sticky',
                    top: '56px',
                    height: 'calc(100vh - 56px)',
                    zIndex: 40
                  }}
                >
                  <Sidebar ref={sidebarRef} onSelect={handleHistorySelect} />
                </motion.div>
              )}
            </AnimatePresence>
            <main className={styles.mainAuth} ref={contentRef}>
              <div className={styles.contentWrapper}>
                <div className={styles.scrollArea}>
                  {showDashboard && <Dashboard />}
                  {result && (
                      <ResultsPanel 
                        query={result.prompt} 
                        result={result} 
                        onClear={() => setResult(null)} 
                      />
                  )}
                </div>
                <div className={styles.promptContainer}>
                  <PromptInput onSubmit={handleQuery} isProcessing={isProcessing} />
                </div>
              </div>
            </main>
          </div>
        ) : (
          <main className={styles.mainPublic}>
            <Hero />
            <div style={{ maxWidth: '1440px', margin: '48px auto', padding: '0 48px', width: '100%' }}>
              <PromptInput onSubmit={handleQuery} isProcessing={isProcessing} />
            </div>
            {result && (
                <div style={{ maxWidth: '1440px', margin: '0 auto 48px', padding: '0 48px', width: '100%' }}>
                    <ResultsPanel 
                    query={result.prompt}
                    result={result} 
                    onClear={() => setResult(null)} 
                    />
                </div>
            )}
            <BentoGrid />
            <PipelineViz />
            <Footer />
          </main>
        )}
        
        <AuthModal 
          isOpen={authModalOpen} 
          initialMode={authMode}
          onClose={() => setAuthModalOpen(false)} 
          onAuth={(u) => {
            setUser(u)
            // Clear any previous result so user sees clean input on login
            setResult(null)
          }} 
        />

        {user && (
          <ProfileModal 
            isOpen={profileModalOpen}
            onClose={() => setProfileModalOpen(false)}
            username={user}
          />
        )}
      </div>
    </ThemeProvider>
  )
}

export default App
