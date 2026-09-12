import csv
import os
import random
from collections.abc import Iterable
from typing import TypedDict

class Demographic(TypedDict):
    category: str
    cohort: str

class SpeakerProfile(TypedDict):
    filename: str
    category: str
    sex: str
    age: int
    country: str
    native_language: str
    cohort: str
    target_demographic: Demographic

class AssistantSpeaker(TypedDict):
    filename: str
    sex: str
    country: str
    category: str
    cohort: str

TOTAL_NATIVE = 195_433_224  # White
TOTAL_AFRICAN = 42_951_595  # Black or African American
TOTAL_INDIAN = 2_442_428  # American Indian and Alaska Native
TOTAL_ASIAN = 22_080_844 # Asian

# Predefined Native English speaker pool for assistant voices
# Selected from Speech Accent Archive: 5 male + 5 female speakers
ASSISTANT_NATIVE_POOL = {
    "male": [
        {"filename": "english127.mp3", "country": "usa"},     # USA male
        {"filename": "english13.mp3", "country": "uk"},       # UK male
        {"filename": "english105.mp3", "country": "canada"},  # Canada male
        {"filename": "english125.mp3", "country": "australia"}, # Australia male
        {"filename": "english116.mp3", "country": "usa"},     # USA male 2
    ],
    "female": [
        {"filename": "english10.mp3", "country": "usa"},      # USA female
        {"filename": "english11.mp3", "country": "uk"},       # UK female
        {"filename": "english114.mp3", "country": "usa"},     # USA female 2
        {"filename": "english117.mp3", "country": "usa"},     # USA female 3
        {"filename": "english12.mp3", "country": "uk"},       # UK female 2
    ],
}

class DemographicSampler:
    def __init__(
        self,
        base_dir: str = "datasets/SpeechAccentArchive",
        category_strategy: str = "origin_country",
        balance_distribution: bool = False,
        with_selected_speakers: bool = False,
    ):
        self.base_dir = base_dir
        self.category_strategy = category_strategy
        self.balance_distribution = balance_distribution
        self.with_selected_speakers = with_selected_speakers
        self.recordings_dir = os.path.join(base_dir, 'recordings')
        self.speakers_csv = os.path.join(base_dir, "speakers_all.csv")
        self.selected_speakers_csv = os.path.join(base_dir, "selected_speakers.csv")

        # Track sampled demographics for balancing
        self._sampled_counts = {
            "sex": {},      # {"male": count, "female": count}
            "cohort": {},   # {"Generation Z": count, ...}
        }
        # Categories map
        self.categories_map = {
            'Native': [
                "usa",
                "uk",
                "canada",
                "australia",
                "ireland",
            ],
            'African': [
                "ethiopia",
                "nigeria",
                "ghana",
                "senegal",
                "morocco",
            ],
            "Indian": [
                "india",
                "pakistan",
                "afghanistan",
                "bangladesh",
                "nepal",
            ],
            'Asian': [
                "china",
                "south korea",
                "japan",
                "philippines",
                "vietnam",
            ]
        }
        
        self.country_to_category = {}
        for cat, countries in self.categories_map.items():
            for c in countries:
                self.country_to_category[c.lower()] = cat

        # Age Cohorts (age-based ranges)
        self.cohorts = {
            '10': (10, 20),      # 10-19 years old
            '20-30': (20, 40),   # 20-39 years old
            '40-50': (40, 60),   # 40-59 years old
            '60+': (60, 200),    # 60+ years old
        }

        self.speakers = self._load_speakers()
        self.category_cohort_counts = self._build_category_cohort_counts()

    def _age_to_cohort(self, age):
        if age is None:
            return None
        for cohort, (min_age, max_age) in self.cohorts.items():
            if min_age <= age < max_age:
                return cohort
        return None

    def _load_speakers(self):
        if self.with_selected_speakers:
            return list(self._iter_selected_speakers())

        speakers = []
        for speaker in self._iter_speakers_from_csv():
            speaker_out = speaker.copy()
            speaker_out["category"] = self._infer_category(speaker_out)
            speaker_out["cohort"] = self._age_to_cohort(speaker_out.get("age"))
            if speaker_out.get("category") and speaker_out.get("cohort"):
                speakers.append(speaker_out)
        return speakers

    def _iter_speakers_from_csv(self) -> Iterable[dict]:
        if not os.path.exists(self.speakers_csv):
            raise FileNotFoundError(
                f"Missing SAA speaker metadata CSV: {self.speakers_csv}"
            )

        with open(self.speakers_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_missing = (row.get("file_missing?") or "").strip().upper()
                if file_missing == "TRUE":
                    continue

                filename_base = (row.get("filename") or "").strip()
                if not filename_base:
                    continue

                filename = f"{filename_base}.mp3"
                if not self._audio_exists(filename):
                    continue

                age_raw = (row.get("age") or "").strip()
                try:
                    age = int(age_raw) if age_raw else None
                except ValueError:
                    age = None

                country = (row.get("country") or "").strip().lower()
                native_language = (row.get("native_language") or "").strip().lower()
                sex = (row.get("sex") or "").strip().lower()

                yield {
                    "filename": filename,
                    "category": None,
                    "sex": sex,
                    "age": age,
                    "country": country,
                    "native_language": native_language,
                    "cohort": None,
                }

    def _iter_selected_speakers(self) -> Iterable[dict]:
        """Load the curated user speaker pool without altering its metadata."""
        if not os.path.exists(self.selected_speakers_csv):
            raise FileNotFoundError(
                "Missing selected speaker manifest: "
                f"{self.selected_speakers_csv}"
            )

        required_fields = {
            "filename",
            "category",
            "cohort",
            "sex",
            "age",
            "country",
            "native_language",
        }
        with open(self.selected_speakers_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing_fields = required_fields - set(reader.fieldnames or [])
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(
                    "Selected speaker manifest is missing required columns: "
                    f"{missing}"
                )

            for row in reader:
                filename = (row["filename"] or "").strip()
                if not filename:
                    raise ValueError("Selected speaker manifest contains an empty filename")
                if not self._audio_exists(filename):
                    raise FileNotFoundError(
                        "Selected speaker audio is missing: "
                        f"{filename} in {self.base_dir}"
                    )

                age_raw = (row["age"] or "").strip()
                try:
                    age = int(age_raw)
                except ValueError as exc:
                    raise ValueError(
                        "Selected speaker manifest contains an invalid age for "
                        f"{filename}: {age_raw!r}"
                    ) from exc

                yield {
                    "filename": filename,
                    "category": (row["category"] or "").strip(),
                    "cohort": (row["cohort"] or "").strip(),
                    "sex": (row["sex"] or "").strip().lower(),
                    "age": age,
                    "country": (row["country"] or "").strip().lower(),
                    "native_language": (row["native_language"] or "").strip().lower(),
                }

    def _audio_exists(self, filename: str) -> bool:
        direct_root = os.path.join(self.base_dir, filename)
        if os.path.exists(direct_root):
            return True
        direct = os.path.join(self.recordings_dir, filename)
        if os.path.exists(direct):
            return True
        # Some datasets nest recordings under recordings/recordings
        nested = os.path.join(self.recordings_dir, "recordings", filename)
        if os.path.exists(nested):
            return True
        for cat in ("Native", "African", "Asian", "Indian"):
            p = os.path.join(self.base_dir, cat, filename)
            if os.path.exists(p):
                return True
        return False

    def _build_category_cohort_counts(self):
        counts = {}
        for speaker in self.speakers:
            category = speaker.get('category')
            cohort = speaker.get('cohort')
            if not category or not cohort:
                continue
            counts.setdefault(category, {})
            counts[category][cohort] = counts[category].get(cohort, 0) + 1
        return counts

    def _infer_category(self, speaker):
        country = (speaker.get('country') or '').lower()
        if self.category_strategy == "origin_country":
            inferred = self.country_to_category.get(country)
            if inferred:
                return inferred
        return speaker.get("category")
    
    def sample_demographic(self) -> Demographic:
        """
        Samples a demographic profile.
        
        Category: weighted by US population counts
        Cohort: uniform distribution (25% each)
        """
        options = []
        # Native (White + American Indian and Alaska Native)
        options.append( ('Native', TOTAL_NATIVE + TOTAL_INDIAN) )
        # African
        options.append( ('African', TOTAL_AFRICAN) )
        # Indian 
        options.append(('Indian', TOTAL_INDIAN))
        # Asian
        options.append( ('Asian', TOTAL_ASIAN) )
                
        total_weight = sum(o[1] for o in options)
        if total_weight == 0: return None
        
        r = random.uniform(0, total_weight)
        cur = 0
        selected = options[-1] 
        for opt in options:
            cur += opt[1]
            if r <= cur:
                selected = opt
                break
        
        target_category, _ = selected

        # Cohort: uniform distribution (not weighted by Archive speaker counts)
        target_cohort = random.choice(list(self.cohorts.keys()))
        
        return {
            'category': target_category,
            'cohort': target_cohort
        }
        
    def find_speaker(self, demographic: Demographic) -> SpeakerProfile:
        """
        Finds a speaker matching Category and Cohort constraints.
        Sex is determined by the found speaker.

        If balance_distribution is True, prefers underrepresented sex/cohort combinations.

        Note: Excludes speakers in ASSISTANT_NATIVE_POOL to prevent
        collision between user and assistant speakers.
        """
        cat = demographic['category']
        cohort = demographic['cohort']

        # Get assistant pool filenames to exclude
        assistant_filenames = set(
            s['filename'] for s in ASSISTANT_NATIVE_POOL['male'] + ASSISTANT_NATIVE_POOL['female']
        )

        # Filter candidates: match category/cohort and exclude assistant pool
        candidates = [
            s for s in self.speakers
            if s['category'] == cat and s['cohort'] == cohort
            and s['filename'] not in assistant_filenames
        ]

        if not candidates:
            # Fallback to category-only match, still excluding assistant pool
            candidates = [
                s for s in self.speakers
                if s['category'] == cat
                and s['filename'] not in assistant_filenames
            ]

        if candidates:
            # Apply balancing logic if enabled
            if self.balance_distribution:
                speaker = self._select_balanced_speaker(candidates)
            else:
                speaker = random.choice(candidates)

            speaker_out = speaker.copy()
            speaker_out['target_demographic'] = demographic

            # Track sampled speaker for balancing
            if self.balance_distribution:
                self._sampled_counts["sex"][speaker["sex"]] = (
                    self._sampled_counts["sex"].get(speaker["sex"], 0) + 1
                )
                self._sampled_counts["cohort"][cohort] = (
                    self._sampled_counts["cohort"].get(cohort, 0) + 1
                )

            return speaker_out

        return None

    def _select_balanced_speaker(self, candidates: list[dict]) -> dict:
        """
        Select speaker from candidates, preferring underrepresented sex.

        Balances sex distribution across all samples.
        """
        # Count available candidates by sex
        by_sex = {"male": [], "female": []}
        for c in candidates:
            sex = c.get("sex", "").lower()
            if sex in by_sex:
                by_sex[sex].append(c)

        # Get current counts
        male_count = self._sampled_counts["sex"].get("male", 0)
        female_count = self._sampled_counts["sex"].get("female", 0)

        # Prefer underrepresented sex
        if male_count < female_count and by_sex["male"]:
            return random.choice(by_sex["male"])
        elif female_count < male_count and by_sex["female"]:
            return random.choice(by_sex["female"])
        else:
            # Equal or no preference: choose randomly
            return random.choice(candidates)

    def get_distribution_stats(self) -> dict:
        """Return current distribution statistics."""
        return {
            "sex": dict(self._sampled_counts["sex"]),
            "cohort": dict(self._sampled_counts["cohort"]),
        }

    def sample_assistant_speaker(self) -> AssistantSpeaker | None:
        """
        Samples an assistant speaker from the predefined Native English pool.

        Note: No collision check needed since find_speaker() already excludes
        the assistant pool when sampling user speakers.

        Returns:
            AssistantSpeaker dict with filename, sex, country, category, and cohort.
        """
        # Build candidate pool from both male and female speakers
        all_candidates = []
        for sex, speakers in ASSISTANT_NATIVE_POOL.items():
            for speaker in speakers:
                filename = speaker["filename"]
                # Find speaker profile in self.speakers to get category and cohort
                speaker_profile = next(
                    (s for s in self.speakers if s["filename"] == filename),
                    None
                )

                if speaker_profile:
                    all_candidates.append({
                        "filename": filename,
                        "sex": sex,
                        "country": speaker["country"],
                        "category": speaker_profile.get("category", "Native"),
                        "cohort": speaker_profile.get("cohort", "Unknown"),
                    })
                else:
                    # Fallback if not found in speakers list
                    all_candidates.append({
                        "filename": filename,
                        "sex": sex,
                        "country": speaker["country"],
                        "category": "Native",
                        "cohort": "Unknown",
                    })

        return random.choice(all_candidates) if all_candidates else None

if __name__ == "__main__":
    sampler = DemographicSampler()
    
    try:
        print("\nSampling 5 user/assistant pairs:")
        for i in range(5):
            # Sample user demographic and find speaker (excludes assistant pool)
            demo = sampler.sample_demographic()
            if demo:
                user = sampler.find_speaker(demo)
                if user:
                    # Sample assistant speaker from Native pool
                    assistant = sampler.sample_assistant_speaker()
                    print(
                        f"[{i+1}] User: {user['filename']} ({demo['category']}, {user['cohort']}, {user['sex']}) | "
                        f"Assistant: {assistant['filename']} ({assistant['category']}, {assistant['cohort']}, {assistant['sex']})"
                    )
    except BrokenPipeError:
        pass
