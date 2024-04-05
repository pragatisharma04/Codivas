package constants;

import java.util.Map;

public class AppSyncRequest {
    private Map<String, Object> arguments;
    private Map<String, Object> identity;
    private String fieldName;

    public Map<String, Object> getArguments() {
        return arguments;
    }

    public Map<String, Object> getIdentity() {
        return identity;
    }

    public String getFieldName() {
        return fieldName;
    }
}
