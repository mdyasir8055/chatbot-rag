import { useEffect, useState } from "react";
import API from "./api";

function App() {
  const [message, setMessage] = useState("");

  useEffect(() => {
    API.get("/")
      .then((res) => setMessage(res.data.message))
      .catch((err) => setMessage("Error: " + err.message));
  }, []);

  return (
    <div className="p-10">
      <h1 className="text-2xl font-bold">Frontend Connected 🚀</h1>
      <p>{message}</p>
    </div>
  );
}

export default App;
