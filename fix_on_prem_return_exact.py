with open('app/services/model_router.py', 'r') as f:
    lines = f.readlines()

# Find the exact location (around line 143-151)
for i in range(len(lines)):
    if 'provider_key = model.provider.value' in lines[i]:
        # Replace the section from this line to the return statement
        # We need to insert the on_prem check between line 144 and 145
        
        # Insert new lines after provider_key assignment
        new_lines = [
            '\n',
            '        # Special handling for on_prem models\n',
            '        if provider_key == "on_prem":\n',
            '            if not settings.ON_PREM_MODEL_ENABLED:\n',
            '                raise HTTPException(\n',
            '                    status_code=400,\n',
            '                    detail=f"On-premises service not configured"\n',
            '                )\n',
            '            return model, self._get_on_prem_service(model)\n',
            '\n'
        ]
        
        # Insert after the provider_key line (i+1)
        lines[i+1:i+1] = new_lines
        print(f"Inserted on_prem handling at line {i+2}")
        break

# Write back
with open('app/services/model_router.py', 'w') as f:
    f.writelines(lines)

print("Fixed on_prem return logic")
