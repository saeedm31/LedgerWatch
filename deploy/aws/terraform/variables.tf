variable "name" {
  description = "Name prefix for every resource."
  type        = string
  default     = "ledgerwatch"
}

variable "environment" {
  description = "dev, staging or prod."
  type        = string
  default     = "dev"
}

variable "region" {
  description = "AWS region. EU by default so ledger data and model calls stay in the EU."
  type        = string
  default     = "eu-central-1"
}

variable "github_repository" {
  description = "owner/repo allowed to deploy through GitHub Actions OIDC."
  type        = string
  default     = "saeedm31/LedgerWatch"
}

variable "create_github_oidc_provider" {
  description = "Create the GitHub OIDC provider. Set false if the account already has one."
  type        = bool
  default     = true
}

variable "image_tag" {
  description = "Image tag in ECR. The deploy workflow replaces it with the commit SHA on every release."
  type        = string
  default     = "latest"
}

# ---------- network ----------
variable "vpc_cidr" {
  type    = string
  default = "10.40.0.0/16"
}

variable "internal_alb" {
  description = "true: the app is reachable only inside the company network (VPN / Direct Connect). Usual for internal tools."
  type        = bool
  default     = false
}

variable "allowed_cidrs" {
  description = "Who may reach the load balancer. Use the company's IP ranges in production."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# ---------- HTTPS and single sign-on ----------
variable "certificate_arn" {
  description = "ACM certificate for HTTPS. Empty: HTTP only, for a first test. Single sign-on needs HTTPS."
  type        = string
  default     = ""
}

variable "oidc" {
  description = "Company single sign-on at the load balancer (Microsoft Entra ID, Okta, ...). null: no login."
  type = object({
    issuer                 = string
    authorization_endpoint = string
    token_endpoint         = string
    user_info_endpoint     = string
    client_id              = string
  })
  default = null
}

variable "oidc_client_secret" {
  description = "Client secret of the SSO app registration."
  type        = string
  default     = ""
  sensitive   = true
}

variable "app_domain" {
  description = "Domain name of the app, e.g. ledgerwatch.example.com. Empty: the load balancer name."
  type        = string
  default     = ""
}

# ---------- app ----------
variable "cpu" {
  description = "Task CPU units (1024 = 1 vCPU), shared by the app and the two MCP servers."
  type        = number
  default     = 1024
}

variable "memory" {
  description = "Task memory in MiB (training on 100k entries needs about 2 GB)."
  type        = number
  default     = 4096
}

variable "llm_provider" {
  description = "bedrock (Claude in the company's AWS account, no API key) or anthropic (API key in Secrets Manager)."
  type        = string
  default     = "bedrock"

  validation {
    condition     = contains(["bedrock", "anthropic"], var.llm_provider)
    error_message = "llm_provider must be bedrock or anthropic."
  }
}

variable "bedrock_model_ids" {
  description = "App model name -> Bedrock model or inference-profile id enabled in the account."
  type        = map(string)
  default     = {}
}

variable "bedrock_regions" {
  description = "Regions Bedrock may serve requests from (EU cross-region inference stays inside these)."
  type        = list(string)
  default     = ["eu-central-1", "eu-west-1", "eu-west-3", "eu-north-1", "eu-south-1", "eu-south-2", "eu-central-2"]
}

# ---------- ledger ----------
variable "ledger_engine" {
  description = "duckdb: the simulated demo ledger on EFS. snowflake: the ledger in Snowflake (fill var.snowflake)."
  type        = string
  default     = "duckdb"

  validation {
    condition     = contains(["duckdb", "snowflake"], var.ledger_engine)
    error_message = "ledger_engine must be duckdb or snowflake."
  }
}

variable "snowflake" {
  description = "Snowflake connection (the private key goes in the secret, not here). Use a role with SELECT only."
  type = object({
    account   = string
    user      = string
    warehouse = string
    database  = string
    schema    = string
    role      = string
  })
  default = {
    account   = ""
    user      = ""
    warehouse = ""
    database  = ""
    schema    = "LEDGERWATCH"
    role      = ""
  }
}

# ---------- monthly close job ----------
variable "close_schedule" {
  description = "When to score the closed month (EventBridge Scheduler expression). Empty: no schedule."
  type        = string
  default     = "cron(0 6 2 * ? *)" # 06:00 on the 2nd of each month
}

variable "close_schedule_timezone" {
  description = "Time zone of close_schedule."
  type        = string
  default     = "Europe/Berlin"
}

# ---------- operations ----------
variable "log_retention_days" {
  description = "Days to keep app and VPC flow logs (one year by default, for audits)."
  type        = number
  default     = 365
}

variable "alb_logs_bucket" {
  description = "S3 bucket for load-balancer access logs, usually the company's central log bucket. Empty: off."
  type        = string
  default     = ""
}

variable "alarm_email" {
  description = "Email for CloudWatch alarms (app down, errors). Empty: no alarms topic subscription."
  type        = string
  default     = ""
}
