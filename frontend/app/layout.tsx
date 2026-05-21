import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Legal Red Flag Scanner | AI Contract Analysis',
  description: 'AI-powered local LLM that scans legal contracts for critical risks.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
