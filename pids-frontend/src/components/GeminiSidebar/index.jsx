import { useState } from 'react'
import styles from './GeminiSidebar.module.css'

const STARTERS = [
  'Summarize the scene',
  'What changed recently?',
  'Which tracks need attention?',
]

export default function GeminiSidebar({ messages = [], busy = false, onSend }) {
  const [draft, setDraft] = useState('')

  function submit(text) {
    const message = text.trim()
    if (!message || busy) return
    setDraft('')
    onSend?.(message)
  }

  function handleSubmit(event) {
    event.preventDefault()
    submit(draft)
  }

  return (
    <aside className={styles.sidebar} aria-label="Scene analyst">
      <div className={styles.header}>
        <div>
          <h2 className={styles.title}>Scene Analyst</h2>
          <p className={styles.subtitle}>Live context from the cameras,</p>
        </div>
      </div>

      <div className={styles.thread}>
        {messages.length ? (
          <div className={styles.messages}>
            {messages.map(message => (
              <div
                className={[
                  styles.message,
                  message.role === 'user' ? styles.userMessage : styles.assistantMessage,
                  message.error ? styles.errorMessage : '',
                ].filter(Boolean).join(' ')}
                key={message.id}
              >
                {message.text}
              </div>
            ))}
            {busy && <div className={`${styles.message} ${styles.assistantMessage}`}>Analyzing scene...</div>}
          </div>
        ) : (
          <div className={styles.promptGroup} aria-label="Suggested scene questions">
            {STARTERS.map(starter => (
              <button
                className={styles.prompt}
                type="button"
                disabled={busy}
                key={starter}
                onClick={() => submit(starter)}
              >
                {starter}
              </button>
            ))}
          </div>
        )}
      </div>

      <form className={styles.composer} onSubmit={handleSubmit}>
        <input
          aria-label="Scene analyst message"
          value={draft}
          disabled={busy}
          onChange={event => setDraft(event.target.value)}
          placeholder="Ask about tracks, counts, or changes..."
        />
        <button type="submit" disabled={busy || !draft.trim()} title="Send message">
          Ask
        </button>
      </form>
    </aside>
  )
}
