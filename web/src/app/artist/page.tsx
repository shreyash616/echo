import { Suspense } from "react";
import { Loader } from "@/components/InputSection";
import ArtistView from "./ArtistView";

// /artist/?name=<artist>. The backend has no artist endpoint, so this is the artist's songs among the
// top 20 search results for the name.
export default function ArtistPage() {
  return (
    <Suspense fallback={<Loader text="Loading artist..." />}>
      <ArtistView />
    </Suspense>
  );
}
