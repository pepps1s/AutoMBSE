package IBDRootSecurityAccessControl {

    package AccessControlPolicy {
        // Define policy for command authorization
        part def AccessControlPolicy {
            attribute policyId: String;
            attribute roleBasedAccess: Boolean;
            attribute multiFactorAuthRequired: Boolean;
        }

        // Define roles involved in the policy
        part def Role {
            attribute roleName: String;
            attribute accessLevel: String;
        }

        // Define command types related to security
        part def CommandType {
            attribute commandName: String;
            attribute isAuthorized: Boolean;
        }
    }

    package AuthorizationDecision {
        // Define the decision to authorize command execution
        part def AuthorizationDecision {
            attribute decisionId: String;
            attribute isAuthorized: Boolean;
            attribute authorizationTime: DateTime;
        }

        // Define path for routing command enablement based on authorization
        port def CommandEnablementPath {
            inout CommandType authorizedCommand: CommandType;
            out AuthorizationDecision authorizationResult;
        }

        part def AuthorizationHandler {
            port inPort: CommandEnablementPath;
            attribute handlerId: String;
        }
    }

    package AuditLogging {
        // Define structure for audit log entries
        part def AuditLog {
            attribute logId: String;
            attribute eventTimestampUtc: DateTime;
            attribute actorId: String;
            attribute eventType: String;
            attribute severityLevel: String;
        }

        // Define audit log store service for retention and search
        part def AuditLogStoreService {
            port inPort: AuditLog;
            attribute storageRetentionDays: Integer;
            attribute searchable: Boolean;
        }

        // Define ports for linking audit log entries to the storage system
        port def AuditLogInPort {
            in AuditLog auditEntry;
        }

        port def AuditLogOutPort {
            out AuditLog auditEntry;
        }
    }

    // Root model including parts and ports to interconnect policy, authorization, and logging
    part def IBDRootSecurityAccessControl {
        part accessControlPolicy: AccessControlPolicy;
        part role: Role;
        part commandType: CommandType;
        part authorizationDecision: AuthorizationDecision;
        part authorizationHandler: AuthorizationHandler;
        part auditLog: AuditLog;
        part auditLogStoreService: AuditLogStoreService;
        
        port outCommandPath: CommandEnablementPath;
        port outAuditLog: AuditLogOutPort;
    }

}
