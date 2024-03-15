function App() {
  return (
    <div className="bg-slate-500 h-screen flex justify-center items-center">
      <div className="flex flex-col items-center gap-4">
        <h1 className="text-white text-xl font-semibold">
          Tell me your plan for the day!
        </h1>
        <input
          type="text"
          className="py-3 px-4 w-full max-w-md border border-black rounded-lg text-sm focus:border-blue-500 focus:ring-blue-500"
        />
        <button
          type="button"
          className="py-3 px-4 flex items-center gap-x-2 text-sm font-semibold rounded-lg bg-blue-100 text-blue-800 hover:bg-blue-200"
        >
          Get your calendar events!
        </button>
      </div>
    </div>
  );
}

export default App;
