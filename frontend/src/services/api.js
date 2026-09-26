const API = import.meta.env.VITE_API_BASE_URL || ''

function getAuthHeaders() {
  const h = { 'Content-Type': 'application/json' }
  const token = localStorage.getItem('cora_token')
  if (token) h['Authorization'] = `Bearer ${token}`
  return h
}

function extractError(detail) {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map(e => `${e.loc[e.loc.length-1]}: ${e.msg}`).join(', ')
  }
  if (detail && typeof detail === 'object') return JSON.stringify(detail)
  return null
}

export async function fetchStats() {
  const res = await fetch(`${API}/v1/stats`)
  return res.json()
}

export async function submitQuery(prompt, userApiKey) {
  const res = await fetch(`${API}/v1/query`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify({ prompt, user_api_key: userApiKey || undefined }),
  })
  return res.json()
}

export async function streamQuery(prompt, userApiKey, onMeta, onToken, onDone, onError) {
  try {
    const res = await fetch(`${API}/v1/query/stream`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ prompt, user_api_key: userApiKey || undefined }),
    })
    
    if (!res.ok) {
      const err = await res.json()
      throw new Error(extractError(err.detail) || 'Streaming failed')
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n\n')
      buffer = lines.pop() || '' // Keep the last incomplete chunk in the buffer

      for (const block of lines) {
        if (!block.trim()) continue
        
        const linesInBlock = block.split('\n')
        let eventType = 'message'
        let dataStr = ''
        
        for (const line of linesInBlock) {
          if (line.startsWith('event:')) {
            eventType = line.replace('event:', '').trim()
          } else if (line.startsWith('data:')) {
            dataStr = line.replace('data:', '').trim()
          }
        }
        
        if (dataStr) {
          try {
            const data = JSON.parse(dataStr)
            if (eventType === 'meta' && onMeta) onMeta(data)
            else if (eventType === 'token' && onToken) onToken(data.text)
            else if (eventType === 'done' && onDone) onDone(data)
            else if (eventType === 'error' && onError) onError(data.error)
          } catch (e) {
            console.error('Failed to parse SSE data', dataStr)
          }
        }
      }
    }
  } catch (err) {
    if (onError) onError(err.message)
  }
}

export async function getCognitiveProfile(prompt) {
  const res = await fetch(`${API}/v1/cognitive-profile`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  })
  return res.json()
}

export async function login(username, password) {
  const res = await fetch(`${API}/v1/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(extractError(err.detail) || 'Login failed')
  }
  return res.json()
}

export async function register(username, email, password) {
  const res = await fetch(`${API}/v1/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, email, password }),
  })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(extractError(err.detail) || 'Registration failed')
  }
  return res.json()
}

export async function logout() {
  await fetch(`${API}/v1/auth/logout`, {
    method: 'POST',
    headers: getAuthHeaders(),
  }).catch(() => {})
}

export async function fetchHistory(page = 1, pageSize = 100) {
  const res = await fetch(`${API}/v1/user/history?page=${page}&page_size=${pageSize}`, {
    headers: getAuthHeaders(),
  })
  if (res.status === 401) {
    return { queries: [], total: 0, page: 1, page_size: pageSize, has_more: false }
  }
  return res.json()
}

export async function deleteHistoryItem(id) {
  const res = await fetch(`${API}/v1/user/history/${id}`, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(extractError(err.detail) || 'Delete failed')
  }
  return res.json()
}

export async function updateProfile(email, password) {
  const payload = {}
  if (email) payload.email = email
  if (password) payload.password = password

  const res = await fetch(`${API}/v1/user/profile`, {
    method: 'PUT',
    headers: getAuthHeaders(),
    body: JSON.stringify(payload)
  })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(extractError(err.detail) || 'Profile update failed')
  }
  return res.json()
}

export async function optimizePrompt(prompt) {
  const res = await fetch(`${API}/v1/optimize`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify({ prompt })
  })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(extractError(err.detail) || 'Optimization failed')
  }
  return res.json()
}
