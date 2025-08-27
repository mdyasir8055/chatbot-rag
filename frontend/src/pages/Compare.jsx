import React, { useState } from 'react';
import { Upload, X, FileText } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const Compare = () => {
  const navigate = useNavigate();
  const [docA, setDocA] = useState(null);
  const [docB, setDocB] = useState(null);
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);

  const onUpload = async (file, which) => {
    try {
      const { uploadDocument } = await import('../api');
      await uploadDocument(file);
      if (which === 'A') setDocA(file.name);
      else setDocB(file.name);
    } catch (e) {
      console.error(e);
      alert('Upload failed');
    }
  };

  const onCompare = async () => {
    if (!docA || !docB || !query.trim()) {
      alert('Please upload Document A and B and enter a query.');
      return;
    }
    setBusy(true);
    try {
      const { compareDocuments } = await import('../api');
      const res = await compareDocuments({ query, docA, docB });
      setResult(res.result || res);
    } catch (e) {
      console.error(e);
      alert('Compare failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Left Panel: Documents & Uploads */}
      <div className="w-[520px] bg-white border-r border-gray-200 flex flex-col">
        {/* Header with Exit Compare */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
              <FileText className="w-5 h-5 text-white" />
            </div>
            <h1 className="font-semibold text-gray-900">Documents</h1>
          </div>
          <button
            onClick={() => navigate('/')} 
            className="px-3 py-1.5 text-sm bg-blue-50 text-blue-700 rounded-full hover:bg-blue-100"
            title="Exit Compare"
          >
            Exit Compare
          </button>
        </div>

        {/* Two upload slots */}
        <div className="p-4 space-y-4 overflow-y-auto">
          <div className="grid grid-cols-2 gap-4">
            {[{label: 'Upload Document A', key: 'A'}, {label: 'Upload Document B', key: 'B'}].map(({label, key}) => (
              <label key={key} className="border-2 border-dashed rounded-lg p-6 text-center text-gray-600 cursor-pointer">
                <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                <div className="font-medium">{label}</div>
                <input type="file" accept="application/pdf" className="hidden" onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0], key)} />
              </label>
            ))}
          </div>

          {/* Selected/Uploaded Items */}
          <div className="space-y-3">
            {docA && (
              <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2">
                    <FileText className="w-4 h-4 text-gray-500" />
                    <div>
                      <div className="text-sm font-medium text-gray-900">{docA}</div>
                      <div className="text-xs text-gray-600">Document A</div>
                    </div>
                  </div>
                  <button className="text-gray-400 hover:text-gray-600" title="Remove" onClick={() => setDocA(null)}>
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
            {docB && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2">
                    <FileText className="w-4 h-4 text-gray-500" />
                    <div>
                      <div className="text-sm font-medium text-gray-900">{docB}</div>
                      <div className="text-xs text-gray-600">Document B</div>
                    </div>
                  </div>
                  <button className="text-gray-400 hover:text-gray-600" title="Remove" onClick={() => setDocB(null)}>
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Right Panel: Comparison */}
      <div className="flex-1 p-6 overflow-y-auto">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-semibold text-gray-900">Document Comparison</h2>
          <button className="px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
            Export Report
          </button>
        </div>

        {/* Query and Compare */}
        <div className="bg-white border border-gray-200 rounded-xl p-5 mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">Comparison Query</label>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., Compare pricing, features, and support"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <div className="mt-3">
            <button onClick={onCompare} disabled={busy} className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50">
              {busy ? 'Comparing...' : 'Compare'}
            </button>
          </div>
        </div>

        {/* Results */}
        {result && (
          <div className="grid grid-cols-2 gap-6">
            <div className="bg-white border border-gray-200 rounded-xl p-5">
              <div className="text-lg font-semibold text-gray-900 mb-2">{docA || 'Document A'}</div>
              <div className="font-medium mb-2">Features</div>
              <ul className="list-disc list-inside text-gray-700 mb-3">
                {(result.features_a || []).map((x, i) => <li key={i}>{x}</li>)}
              </ul>
              <div className="text-green-700 font-medium mb-1">Pros</div>
              <ul className="list-disc list-inside text-green-700 mb-3">
                {(result.pros_a || []).map((x, i) => <li key={i}>{x}</li>)}
              </ul>
              <div className="text-red-700 font-medium mb-1">Cons</div>
              <ul className="list-disc list-inside text-red-700">
                {(result.cons_a || []).map((x, i) => <li key={i}>{x}</li>)}
              </ul>
            </div>
            <div className="bg-white border border-gray-200 rounded-xl p-5">
              <div className="text-lg font-semibold text-gray-900 mb-2">{docB || 'Document B'}</div>
              <div className="font-medium mb-2">Features</div>
              <ul className="list-disc list-inside text-gray-700 mb-3">
                {(result.features_b || []).map((x, i) => <li key={i}>{x}</li>)}
              </ul>
              <div className="text-green-700 font-medium mb-1">Pros</div>
              <ul className="list-disc list-inside text-green-700 mb-3">
                {(result.pros_b || []).map((x, i) => <li key={i}>{x}</li>)}
              </ul>
              <div className="text-red-700 font-medium mb-1">Cons</div>
              <ul className="list-disc list-inside text-red-700">
                {(result.cons_b || []).map((x, i) => <li key={i}>{x}</li>)}
              </ul>
            </div>
            {result.summary && (
              <div className="col-span-2 bg-white border border-gray-200 rounded-xl p-5">
                <div className="text-lg font-semibold text-gray-900 mb-2">Summary</div>
                <p className="text-gray-700 whitespace-pre-wrap">{result.summary}</p>
              </div>
            )}
            {result.raw && (
              <div className="col-span-2 bg-white border border-gray-200 rounded-xl p-5">
                <div className="text-lg font-semibold text-gray-900 mb-2">Model Output</div>
                <pre className="text-sm text-gray-700 overflow-auto">{result.raw}</pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Compare;