// Mirrors TrackResult in backend/app/models/schemas.py (the fields the web app reads).
export interface Track {
  id: string;
  title: string;
  artist: string;
  album?: string;
  albumArtUrl?: string;
  previewUrl?: string | null;
  matchScore?: number | null;
  vibes?: string[];
  felt_bpm?: number | null;
  key?: string;
  theme?: string | null;
  energy?: number | null;
  valence?: number | null;
  languageAffinity?: number | null;
  year?: number | null;
  decade?: string | null;
  matchReasons?: string[];
  instruments?: string[];
  spatialStaging?: string;
  lyricalIntent?: string;
}

export interface IdentifyResponse {
  identified: Track | null;
  track?: Track | null;
  recommendations: Track[];
}

export interface ClientConfig {
  spotify_client_id?: string;
  spotify_redirect_uri?: string;
}
