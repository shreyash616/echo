import { Suspense } from "react";
import { Loader } from "@/components/InputSection";
import SearchView from "./SearchView";

// /search/?q=<query>: up to 20 songs, the most /api/search returns
export default function SearchPage() {
  return (
    <Suspense fallback={<Loader text="Searching songs..." />}>
      <SearchView />
    </Suspense>
  );
}
