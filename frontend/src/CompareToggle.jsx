import React from 'react';
import { useNavigate } from 'react-router-dom';

// Simple controlled checkbox that navigates when enabled
export default function CompareToggle({ checked, onChange }) {
  const navigate = useNavigate();

  const handleChange = (e) => {
    const val = e.target.checked;
    onChange?.(val);
    if (val) navigate('/compare');
  };

  return (
    <label className="flex items-center gap-2 text-sm text-gray-600">
      <input
        type="checkbox"
        checked={checked}
        onChange={handleChange}
        className="rounded border-gray-300"
      />
      Compare Mode
    </label>
  );
}