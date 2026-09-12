#!/usr/bin/env bash
set -euo pipefail

DATASET_SPEC=""
DATA_DIR="${DATA_DIR:-datasets}"
FORCE="${FORCE:-0}"
BACKGROUND="${BACKGROUND:-0}"
LOG_FILE="${LOG_FILE:-download_dataset.log}"

# Default sources
EMOWOZ_URL_DEFAULT="https://zenodo.org/records/14810836/files/dst.zip?download=1"
SGD_REPO_DEFAULT="https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git"
ABCD_REPO_DEFAULT="https://github.com/asappresearch/abcd.git"
MULTIWOZ_REPO_DEFAULT="https://github.com/budzianowski/multiwoz.git"
TM2_REPO_DEFAULT="https://github.com/google-research-datasets/Taskmaster.git"
SPOKENWOZ_TRAIN_AUDIO_URL_DEFAULT="https://huggingface.co/datasets/ssz1111/SpokenWOZ-Train-Audio/resolve/main/audio_5700_train_dev.tar.gz"
SPOKENWOZ_TRAIN_TEXT_URL_DEFAULT="https://huggingface.co/datasets/ssz1111/SpokenWOZ-Train-Text/resolve/main/data.json?download=true"
SPOKENWOZ_TRAIN_VALLIST_URL_DEFAULT="https://huggingface.co/datasets/ssz1111/SpokenWOZ-Train-Text/resolve/main/valListFile.json?download=true"
SPOKENWOZ_TEST_AUDIO_URL_DEFAULT="https://huggingface.co/datasets/ssz1111/SpokenWOZ-Test-Audio-Fixed/resolve/main/audio_5700_test.zip?download=true"
SPOKENWOZ_TEST_TEXT_URL_DEFAULT="https://huggingface.co/datasets/ssz1111/SpokenWOZ-Test-Text-Fixed/resolve/main/data.json?download=true"
SPOKENWOZ_TEST_LIST_URL_DEFAULT="https://huggingface.co/datasets/ssz1111/SpokenWOZ-Test-Text-Fixed/resolve/main/testListFile.json?download=true"
SAA_URL_DEFAULT="https://www.kaggle.com/api/v1/datasets/download/rtatman/speech-accent-archive"

usage() {
  cat <<EOF
Usage: scripts/download_dataset.sh <dataset[,dataset...]>

Datasets:
  emowoz   (requires MultiWOZ)
  sgd
  abcd
  tm2
  spokenwoz
  multiwoz
  saa      (SpeechAccentArchive)
  all

Env vars (optional):
  DATASETS              comma-separated dataset list (alternative to args)
  DATA_DIR              base datasets dir (default: datasets)
  EMOWOZ_URL            archive URL
  MULTIWOZ_REPO         git repo URL
  SGD_REPO              git repo URL
  ABCD_REPO             git repo URL
  TM2_REPO              git repo URL
  SPOKENWOZ_TRAIN_AUDIO_URL  train audio tar.gz URL
  SPOKENWOZ_TRAIN_TEXT_URL   train text data.json URL
  SPOKENWOZ_TRAIN_VALLIST_URL valListFile.json URL
  SPOKENWOZ_TEST_AUDIO_URL   test audio zip URL
  SPOKENWOZ_TEST_TEXT_URL    test text data.json URL
  SPOKENWOZ_TEST_LIST_URL    testListFile.json URL
  SAA_URL               archive URL
  FORCE=1               overwrite existing dataset dir (otherwise error if exists)
  PARALLEL=1            run dataset downloads in parallel
  JOBS=N                max parallel jobs (default: nproc or 6)
  BACKGROUND=1          run in background and print PID/kill command
  LOG_FILE=path         background log file (default: download_dataset.log)

Default sources (Can be changed):
  EMOWOZ_URL=${EMOWOZ_URL_DEFAULT}
  MULTIWOZ_REPO=${MULTIWOZ_REPO_DEFAULT}
  SGD_REPO=${SGD_REPO_DEFAULT}
  ABCD_REPO=${ABCD_REPO_DEFAULT}
  TM2_REPO=${TM2_REPO_DEFAULT}
  SPOKENWOZ_TRAIN_AUDIO_URL=${SPOKENWOZ_TRAIN_AUDIO_URL_DEFAULT}
  SPOKENWOZ_TRAIN_TEXT_URL=${SPOKENWOZ_TRAIN_TEXT_URL_DEFAULT}
  SPOKENWOZ_TRAIN_VALLIST_URL=${SPOKENWOZ_TRAIN_VALLIST_URL_DEFAULT}
  SPOKENWOZ_TEST_AUDIO_URL=${SPOKENWOZ_TEST_AUDIO_URL_DEFAULT}
  SPOKENWOZ_TEST_TEXT_URL=${SPOKENWOZ_TEST_TEXT_URL_DEFAULT}
  SPOKENWOZ_TEST_LIST_URL=${SPOKENWOZ_TEST_LIST_URL_DEFAULT}
  SAA_URL=${SAA_URL_DEFAULT}
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

if [[ "$BACKGROUND" == "1" || "$BACKGROUND" == "true" ]]; then
  if [[ -z "${BACKGROUND_CHILD:-}" ]]; then
    if command -v nohup >/dev/null 2>&1; then
      nohup env BACKGROUND_CHILD=1 BACKGROUND=0 bash "$0" "$@" >"$LOG_FILE" 2>&1 &
    else
      env BACKGROUND_CHILD=1 BACKGROUND=0 bash "$0" "$@" >"$LOG_FILE" 2>&1 &
    fi
    pid="$!"
    echo "Started background download (pid=$pid). Logs: $LOG_FILE"
    echo "To stop: kill $pid"
    exit 0
  fi
fi

ensure_dir() {
  mkdir -p "$1"
}

ensure_git() {
  if ! command -v git >/dev/null 2>&1; then
    die "git is required to clone repositories."
  fi
}

download_file() {
  local url="$1"
  local dest="$2"

  log "Downloading: $url"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --progress-bar --show-error "$url" -o "$dest"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$dest" "$url"
  else
    die "curl or wget is required to download."
  fi
  log "Downloaded: $dest"
}

download_and_extract() {
  local url="$1"
  local dest_dir="$2"
  local tmpdir
  tmpdir="$(mktemp -d)"
  local filename="${tmpdir}/archive"

  download_file "$url" "$filename"


  local url_base="${url%%\?*}"
  

  if [[ "$url_base" == *.zip ]] || [[ "$url" == *"speech-accent-archive"* ]]; then
    ensure_dir "$dest_dir"
    unzip -q "$filename" -d "$dest_dir"
  elif [[ "$url_base" == *.tar.gz || "$url_base" == *.tgz ]]; then
    ensure_dir "$dest_dir"
    tar -xzf "$filename" -C "$dest_dir"
  else

    local file_type
    file_type=$(file -b --mime-type "$filename")
    if [[ "$file_type" == "application/zip" ]]; then
        ensure_dir "$dest_dir"
        unzip -q "$filename" -d "$dest_dir"
    else
        rm -rf "$tmpdir"
        die "Unknown archive type for $url (expected .zip or .tar.gz)."
    fi
  fi

  rm -rf "$tmpdir"
}

maybe_clear_dir() {
  local dir="$1"
  if [[ -d "$dir" ]]; then
    if [[ "$FORCE" == "1" ]]; then
      rm -rf "$dir"
    else
      die "Target datasets exist: $dir (set FORCE=1 to overwrite)"
    fi
  fi
}

require_url() {
  local name="$1"
  local url="$2"
  if [[ -z "$url" ]]; then
    die "Missing ${name}_URL. Provide the dataset archive URL via env var."
  fi
}

require_repo() {
  local name="$1"
  local url="$2"
  if [[ -z "$url" ]]; then
    die "Missing ${name}_REPO. Provide the git repo URL via env var."
  fi
}

link_audio_files() {
  local src_dir="$1"
  local dest_dir="$2"

  if [[ ! -d "$src_dir" ]]; then
    return
  fi

  ensure_dir "$dest_dir"

  while IFS= read -r -d '' file; do
    local base
    base="$(basename "$file")"
    local target="$dest_dir/$base"
    if [[ -e "$target" ]]; then
      continue
    fi
    if ln "$file" "$target" 2>/dev/null; then
      :
    elif ln -s "$file" "$target" 2>/dev/null; then
      :
    else
      cp "$file" "$target"
    fi
  done < <(find "$src_dir" -type f -name "*.wav" -print0)
}

download_emowoz() {
  ensure_dir "$DATA_DIR"
  maybe_clear_dir "$DATA_DIR/EmoWOZ"
  local url="${EMOWOZ_URL:-$EMOWOZ_URL_DEFAULT}"
  require_url "EMOWOZ" "$url"

  local tmpdir
  tmpdir="$(mktemp -d)"
  local archive="${tmpdir}/emowoz.zip"
  download_file "$url" "$archive"
  unzip -q "$archive" -d "$tmpdir/extract"

  local src_dir="$tmpdir/extract/dst/data"
  if [[ ! -d "$src_dir" ]]; then
    rm -rf "$tmpdir"
    die "EmoWOZ archive format unexpected (missing dst/data)."
  fi

  ensure_dir "$DATA_DIR/EmoWOZ"
  shopt -s dotglob
  mv "$src_dir"/* "$DATA_DIR/EmoWOZ"/
  shopt -u dotglob
  rm -rf "$tmpdir"

  # MultiWOZ is required by loader
  if [[ ! -f "$DATA_DIR/MultiWOZ_2.1/data.json" && ! -f "$DATA_DIR/MultiWOZ_2.1/data.json" ]]; then
    echo "Note: EmoWOZ requires MultiWOZ. Run: scripts/download_dataset.sh multiwoz"
  fi
}

download_multiwoz() {
  ensure_dir "$DATA_DIR"
  ensure_git
  local repo="${MULTIWOZ_REPO:-$MULTIWOZ_REPO_DEFAULT}"
  require_repo "MULTIWOZ" "$repo"
  maybe_clear_dir "$DATA_DIR/MultiWOZ_2.1"

  local tmpdir
  tmpdir="$(mktemp -d)"
  
  # Sparse checkout으로 필요한 파일만 가져오기
  git clone --filter=blob:none --sparse "$repo" "$tmpdir/multiwoz"
  (cd "$tmpdir/multiwoz" && git sparse-checkout set data/MultiWOZ_2.1.zip)
  (cd "$tmpdir/multiwoz" && git checkout)

  # MultiWOZ_2.1.zip 파일 확인 및 압축 해제
  local zip_file="$tmpdir/multiwoz/data/MultiWOZ_2.1.zip"
  if [[ ! -f "$zip_file" ]]; then
    rm -rf "$tmpdir"
    die "MultiWOZ_2.1.zip not found in repository"
  fi

  log "Extracting MultiWOZ_2.1.zip..."
  unzip -q "$zip_file" -d "$tmpdir/extracted"

  # 압축 해제된 디렉토리 찾기
  local source_dir=""
  if [[ -d "$tmpdir/extracted/MultiWOZ_2.1" ]]; then
    source_dir="$tmpdir/extracted/MultiWOZ_2.1"
  elif [[ -d "$tmpdir/extracted" ]]; then
    source_dir="$tmpdir/extracted"
  else
    rm -rf "$tmpdir"
    die "Failed to extract MultiWOZ_2.1.zip"
  fi

  ensure_dir "$DATA_DIR/MultiWOZ_2.1"
  shopt -s dotglob
  cp -R "$source_dir"/* "$DATA_DIR/MultiWOZ_2.1"/
  shopt -u dotglob
  rm -rf "$tmpdir"
  
  log "MultiWOZ_2.1 downloaded and extracted successfully"
}

download_sgd() {
  ensure_dir "$DATA_DIR"
  ensure_git
  local repo="${SGD_REPO:-$SGD_REPO_DEFAULT}"
  require_repo "SGD" "$repo"
  maybe_clear_dir "$DATA_DIR/dstc8-schema-guided-dialogue"
  git clone "$repo" "$DATA_DIR/dstc8-schema-guided-dialogue"
}

download_abcd() {
  ensure_dir "$DATA_DIR"
  ensure_git
  local repo="${ABCD_REPO:-$ABCD_REPO_DEFAULT}"
  require_repo "ABCD" "$repo"
  maybe_clear_dir "$DATA_DIR/abcd"
  git clone "$repo" "$DATA_DIR/abcd"

  (cd "$DATA_DIR/abcd/data" && gzip -d abcd_v1.1.json.gz)
}

download_tm2() {
  ensure_dir "$DATA_DIR"
  ensure_git
  
  local repo="${TM2_REPO:-$TM2_REPO_DEFAULT}"
  
  require_repo "TM2" "$repo"
  maybe_clear_dir "$DATA_DIR/TM-2-2020"

  local tmpdir
  tmpdir="$(mktemp -d)"
  
  git clone --filter=blob:none --sparse "$repo" "$tmpdir/taskmaster"
  
  (cd "$tmpdir/taskmaster" && git sparse-checkout set TM-2-2020)

  ensure_dir "$DATA_DIR/TM-2-2020"
  shopt -s dotglob
  cp -R "$tmpdir/taskmaster/TM-2-2020"/* "$DATA_DIR/TM-2-2020"/
  shopt -u dotglob
  
  # 5. 임시 디렉토리 삭제
  rm -rf "$tmpdir"

  # 6. (선택 사항) 데이터 확인 메시지
  if [[ -d "$DATA_DIR/TM-2-2020/data" ]]; then
    echo "Taskmaster-2 data successfully downloaded to $DATA_DIR/TM-2-2020"
  fi
}

download_spokenwoz() {
  ensure_dir "$DATA_DIR"
  maybe_clear_dir "$DATA_DIR/SpokenWOZ"
  local root_dir="$DATA_DIR/SpokenWOZ"
  ensure_dir "$root_dir"

  local train_audio_url="${SPOKENWOZ_TRAIN_AUDIO_URL:-$SPOKENWOZ_TRAIN_AUDIO_URL_DEFAULT}"
  local train_text_url="${SPOKENWOZ_TRAIN_TEXT_URL:-$SPOKENWOZ_TRAIN_TEXT_URL_DEFAULT}"
  local train_vallist_url="${SPOKENWOZ_TRAIN_VALLIST_URL:-$SPOKENWOZ_TRAIN_VALLIST_URL_DEFAULT}"
  local test_audio_url="${SPOKENWOZ_TEST_AUDIO_URL:-$SPOKENWOZ_TEST_AUDIO_URL_DEFAULT}"
  local test_text_url="${SPOKENWOZ_TEST_TEXT_URL:-$SPOKENWOZ_TEST_TEXT_URL_DEFAULT}"
  local test_list_url="${SPOKENWOZ_TEST_LIST_URL:-$SPOKENWOZ_TEST_LIST_URL_DEFAULT}"

  require_url "SPOKENWOZ_TRAIN_AUDIO" "$train_audio_url"
  require_url "SPOKENWOZ_TRAIN_TEXT" "$train_text_url"
  require_url "SPOKENWOZ_TRAIN_VALLIST" "$train_vallist_url"
  require_url "SPOKENWOZ_TEST_AUDIO" "$test_audio_url"
  require_url "SPOKENWOZ_TEST_TEXT" "$test_text_url"
  require_url "SPOKENWOZ_TEST_LIST" "$test_list_url"

  local train_dir="$root_dir/train"
  local test_dir="$root_dir/test"
  local audio_dir="$root_dir/audio"
  local failed=0
  local i
  local -a pids=()
  local -a labels=()
  ensure_dir "$train_dir"
  ensure_dir "$test_dir"
  ensure_dir "$audio_dir"

  download_and_extract "$train_audio_url" "$train_dir" &
  pids+=("$!")
  labels+=("SpokenWOZ train audio")

  download_and_extract "$test_audio_url" "$test_dir" &
  pids+=("$!")
  labels+=("SpokenWOZ test audio")

  download_file "$train_text_url" "$root_dir/train.json" &
  pids+=("$!")
  labels+=("SpokenWOZ train.json")

  download_file "$train_vallist_url" "$root_dir/valListFile.json" &
  pids+=("$!")
  labels+=("SpokenWOZ valListFile.json")

  download_file "$test_text_url" "$root_dir/test.json" &
  pids+=("$!")
  labels+=("SpokenWOZ test.json")

  download_file "$test_list_url" "$root_dir/testListFile.json" &
  pids+=("$!")
  labels+=("SpokenWOZ testListFile.json")

  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      log "Failed: ${labels[$i]}"
      failed=1
    fi
  done

  if [[ "$failed" -ne 0 ]]; then
    die "SpokenWOZ download failed."
  fi

  link_audio_files "$train_dir" "$audio_dir"
  link_audio_files "$test_dir" "$audio_dir"
}

download_saa() {
  local target_dir="$DATA_DIR/SpeechAccentArchive"
  ensure_dir "$DATA_DIR"
  
  maybe_clear_dir "$target_dir"
  
  local url="${SAA_URL:-$SAA_URL_DEFAULT}"
  require_url "SAA" "$url"
  
  download_and_extract "$url" "$target_dir"
}

download_all() {
  download_emowoz
  download_multiwoz
  download_sgd
  download_abcd
  download_tm2
  download_spokenwoz
  download_saa
}

download_one() {
  local dataset="$1"
  log "Dataset: $dataset"
  case "$dataset" in
    emowoz) download_emowoz ;;
    multiwoz) download_multiwoz ;;
    sgd) download_sgd ;;
    abcd) download_abcd ;;
    tm2) download_tm2 ;;
    spokenwoz) download_spokenwoz ;;
    saa) download_saa ;;
    all) download_all ;;
    *) die "Unknown dataset: $dataset" ;;
  esac
}

if [[ "$#" -gt 0 ]]; then
  DATASET_SPEC="$*"
else
  DATASET_SPEC="${DATASETS:-}"
fi

if [[ -z "$DATASET_SPEC" || "$DATASET_SPEC" == "-h" || "$DATASET_SPEC" == "--help" ]]; then
  usage
  exit 0
fi

DATASET_SPEC="${DATASET_SPEC//,/ }"
read -r -a DATASET_LIST <<< "$DATASET_SPEC"

for dataset in "${DATASET_LIST[@]}"; do
  if [[ "$dataset" == "all" ]]; then
    DATASET_LIST=(emowoz multiwoz sgd abcd tm2 spokenwoz saa)
    break
  fi
done

if [[ "${PARALLEL:-1}" == "1" || "${PARALLEL:-1}" == "true" ]]; then
  if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3) )); then
    log "Parallel requested but bash < 4.3; running sequential."
  else
    jobs="${JOBS:-0}"
    if [[ "$jobs" -le 0 ]]; then
      if command -v nproc >/dev/null 2>&1; then
        jobs="$(nproc)"
      else
        jobs=6
      fi
    fi
    if [[ "$jobs" -lt 1 ]]; then
      jobs=1
    fi

    log "Parallel download enabled (jobs=$jobs)"
    running=0
    failed=0
    for dataset in "${DATASET_LIST[@]}"; do
      (download_one "$dataset") &
      ((running+=1))
      if [[ "$running" -ge "$jobs" ]]; then
        if ! wait -n; then
          failed=1
        fi
        ((running-=1))
      fi
    done

    while [[ "$running" -gt 0 ]]; do
      if ! wait -n; then
        failed=1
      fi
      ((running-=1))
    done

    if [[ "$failed" -ne 0 ]]; then
      die "One or more dataset downloads failed."
    fi
    exit 0
  fi
fi

for dataset in "${DATASET_LIST[@]}"; do
  download_one "$dataset"
done
