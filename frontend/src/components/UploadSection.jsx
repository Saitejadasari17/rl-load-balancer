import React, { useState } from 'react';
import { Upload, CheckCircle } from 'lucide-react';

export default function UploadSection() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);

  const handleUpload = async (e) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      const res = await fetch(`${apiUrl}/upload-dataset`, {
        method: 'POST',
        body: formData
      });

      const data = await res.json();
      setFile(selectedFile);
      setUploadResult(data);
    } catch (err) {
      console.error('Upload error:', err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
      <h3 className="font-semibold mb-4 flex items-center gap-2">
        <Upload className="w-5 h-5 text-blue-500" />
        Upload Dataset
      </h3>

      <div className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center hover:border-blue-500 transition cursor-pointer"
        onClick={() => document.getElementById('file-input').click()}>
        <input
          id="file-input"
          type="file"
          accept=".csv"
          onChange={handleUpload}
          disabled={uploading}
          className="hidden"
        />
        <Upload className="w-8 h-8 text-slate-500 mx-auto mb-2" />
        <p className="text-slate-400">Drag and drop your CSV file here</p>
        <p className="text-xs text-slate-500 mt-1">or click to browse</p>
      </div>

      {uploadResult && (
        <div className="mt-4 p-3 bg-green-900/20 border border-green-700 rounded flex items-gap-2">
          <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0 mr-2" />
          <div className="text-sm">
            <p className="font-semibold text-green-300">{uploadResult.filename}</p>
            <p className="text-green-200">{uploadResult.rows} rows loaded</p>
          </div>
        </div>
      )}
    </div>
  );
}
