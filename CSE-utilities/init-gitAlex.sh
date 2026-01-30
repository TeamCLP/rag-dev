#!/bin/bash

# Script de déploiement RAG-DEV
# Date: $(date +"%Y-%m-%d %H:%M:%S")

# Configuration des logs
LOG_FILE="/var/log/rag-dev-setup.log"
SCRIPT_NAME="rag-dev-setup"

# Fonction de logging
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

# Fonction de gestion d'erreur
handle_error() {
    log "ERROR" "Échec de l'exécution à l'étape: $1"
    log "ERROR" "Code de sortie: $2"
    exit $2
}

# Début du script
log "INFO" "=== Début du déploiement RAG-DEV ==="
log "INFO" "Script exécuté par: $(whoami)"
log "INFO" "Répertoire de travail initial: $(pwd)"

# Étape 1: Navigation vers /home
log "INFO" "Navigation vers le répertoire /home"
cd /home || handle_error "Navigation vers /home" $?
log "SUCCESS" "Navigation vers /home réussie"

# Étape 2: Création du répertoire alex
log "INFO" "Création du répertoire alex"
mkdir -p alex || handle_error "Création du répertoire alex" $?
log "SUCCESS" "Répertoire alex créé avec succès"

# Étape 3: Navigation vers le répertoire alex
log "INFO" "Navigation vers le répertoire alex"
cd alex || handle_error "Navigation vers alex" $?
log "SUCCESS" "Navigation vers alex réussie"
log "INFO" "Répertoire de travail actuel: $(pwd)"

# Étape 4: Clonage du repository Git
log "INFO" "Clonage du repository rag-dev depuis GitHub"
if [ -d "rag-dev" ]; then
    log "WARNING" "Le répertoire rag-dev existe déjà, suppression en cours"
    rm -rf rag-dev
fi

git clone https://github.com/TeamCLP/rag-dev.git || handle_error "Clonage du repository" $?
log "SUCCESS" "Clonage du repository réussi"

# Étape 5: Navigation vers le répertoire rag-dev
log "INFO" "Navigation vers le répertoire rag-dev"
cd /home/alex/rag-dev || handle_error "Navigation vers rag-dev" $?
log "SUCCESS" "Navigation vers rag-dev réussie"
log "INFO" "Répertoire final: $(pwd)"

# Vérification du contenu
log "INFO" "Contenu du répertoire rag-dev:"
ls -la | while read line; do
    log "INFO" "  $line"
done

# Fin du script
log "INFO" "=== Déploiement RAG-DEV terminé avec succès ==="
log "INFO" "Répertoire de destination: $(pwd)"

exit 0