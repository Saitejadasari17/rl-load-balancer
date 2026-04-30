import React, { useState, useEffect } from 'react';
import { Activity, TrendingDown, BarChart3, Zap } from 'lucide-react';
import UploadSection from './components/UploadSection';
import TrainingProgress from './components/TrainingProgress';
import MetricCards from './components/MetricCards';
import ResultCharts from './components/ResultCharts';

export default function App() {
  const [apiUrl] = useState(process.env.REACT_APP_API_URL || 'http://localhost:8000');
  const [trainingStatus, setTrainingStatus] = useState('idle');
  const [results, setResults] = useState(null);
  const [trainConfig, setTrainConfig] = useState({
    timesteps: 50000,
    episodes: 10,
    servers: 3
  });

  // Poll training status
  useEffect(() => {
    if (trainingStatus !== 'idle' && trainingStatus !== 'completed') {
      const interval = setInterval(async () => {
        try {
          const res = await fetch(`${apiUrl}/training-status`);
          const data = await res.json();
          
          if (data.status === 'completed') {
            setTrainingStatus('completed');
            fetchResults();
          }
        } catch (err) {
          console.error('Status fetch error:', err);
        }
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [trainingStatus, apiUrl]);

  const fetchResults = async () => {
    try {
      const res = await fetch(`${apiUrl}/results`);
      const data = await res.json();
      setResults(data);
    } catch (err) {
      console.error('Results fetch error:', err);
    }
  };

  const handleStartTraining = async () => {
    setTrainingStatus('training');
    try {
      const res = await fetch(`${apiUrl}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trainConfig)
      });
      
      if (!res.ok) throw new Error('Training failed');
      
      const data = await res.json();
      setResults(data);
      setTrainingStatus('completed');
    } catch (err) {
      console.error('Training error:', err);
      setTrainingStatus('idle');
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center gap-3 mb-2">
            <Zap className="w-8 h-8 text-blue-500" />
            <h1 className="text-3xl font-bold">RL Load Balancer</h1>
          </div>
          <p className="text-slate-400">Adaptive load balancing using Deep Reinforcement Learning</p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        
        {/* Upload & Config Section */}
        <section className="grid md:grid-cols-3 gap-6">
          <div className="md:col-span-2">
            <UploadSection />
          </div>
          <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-green-500" />
              Configuration
            </h3>
            <div className="space-y-4">
              <div>
                <label className="text-sm text-slate-400">Timesteps</label>
                <input 
                  type="number" 
                  value={trainConfig.timesteps}
                  onChange={e => setTrainConfig({...trainConfig, timesteps: parseInt(e.target.value)})}
                  className="w-full mt-1 bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white"
                />
              </div>
              <div>
                <label className="text-sm text-slate-400">Episodes</label>
                <input 
                  type="number" 
                  value={trainConfig.episodes}
                  onChange={e => setTrainConfig({...trainConfig, episodes: parseInt(e.target.value)})}
                  className="w-full mt-1 bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white"
                />
              </div>
              <button
                onClick={handleStartTraining}
                disabled={trainingStatus === 'training'}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 px-4 py-2 rounded font-semibold transition"
              >
                {trainingStatus === 'training' ? 'Training...' : 'Start Training'}
              </button>
            </div>
          </div>
        </section>

        {/* Training Progress */}
        {trainingStatus !== 'idle' && (
          <TrainingProgress status={trainingStatus} />
        )}

        {/* Results */}
        {results && (
          <>
            <MetricCards results={results} />
            <ResultCharts results={results} />
          </>
        )}

        {/* Empty State */}
        {!results && trainingStatus === 'idle' && (
          <div className="text-center py-12 bg-slate-800/30 rounded-lg border border-slate-700">
            <BarChart3 className="w-12 h-12 text-slate-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-slate-400 mb-2">Ready to train</h3>
            <p className="text-slate-500">Upload a dataset and start training your RL agent</p>
          </div>
        )}
      </main>
    </div>
  );
}
