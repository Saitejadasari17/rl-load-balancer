import React, { useEffect, useState } from 'react';
import { Loader } from 'lucide-react';

export default function TrainingProgress({ status }) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress(prev => {
        if (status === 'training') {
          return Math.min(prev + Math.random() * 15, 85);
        }
        return 100;
      });
    }, 500);
    return () => clearInterval(interval);
  }, [status]);

  return (
    <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold flex items-center gap-2">
          <Loader className="w-5 h-5 text-blue-500 animate-spin" />
          Training in Progress
        </h3>
        <span className="text-sm text-slate-400">{Math.floor(progress)}%</span>
      </div>

      <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
        <div
          className="bg-gradient-to-r from-blue-500 to-cyan-500 h-full rounded-full transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>

      <p className="text-sm text-slate-400 mt-3">
        {status === 'training' ? 'Evaluating baselines and training RL agent...' : 'Finalizing results...'}
      </p>
    </div>
  );
}
