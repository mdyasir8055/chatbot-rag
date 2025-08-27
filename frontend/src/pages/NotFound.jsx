import React from 'react';
import { Link } from 'react-router-dom';

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="bg-white p-8 rounded-lg shadow border border-gray-200">
        <h1 className="text-2xl font-semibold mb-2">Page not found</h1>
        <p className="text-gray-600 mb-4">The page you are looking for doesn't exist.</p>
        <Link className="text-blue-600 hover:underline" to="/">Go to Home</Link>
      </div>
    </div>
  );
}