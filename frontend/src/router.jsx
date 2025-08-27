import React from 'react';
import { createBrowserRouter } from 'react-router-dom';
import App from './App.jsx';
import Compare from './pages/Compare.jsx';
import NotFound from './pages/NotFound.jsx';

export const router = createBrowserRouter([
  { path: '/', element: <App /> },
  { path: '/compare', element: <Compare /> },
  { path: '*', element: <NotFound /> },
]);