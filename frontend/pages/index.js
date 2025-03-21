import Head from 'next/head'
import SpamClassifier from '@/components/SpamClassifier'

export default function Home() {
  return (
    <div>
      <Head>
        <title>SMS Spam Detection with Reinforcement Learning</title>
        <meta name="description" content="A spam detection system using Q-learning" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="min-h-screen bg-gray-50">
        <div className="container mx-auto py-8">
          <h1 className="text-3xl font-bold text-center mb-8">
            SMS Spam Detection
          </h1>
          <SpamClassifier />
        </div>
      </main>

      <footer className="bg-white border-t py-6">
        <div className="container mx-auto text-center text-gray-500">
          <p>Spam Detection with Reinforcement Learning &copy; {new Date().getFullYear()}</p>
        </div>
      </footer>
    </div>
  )
}
