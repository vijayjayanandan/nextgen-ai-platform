"""
Test PostgreSQL connection with different passwords
"""

import asyncio
import asyncpg

async def test_postgres_passwords():
    """Test different PostgreSQL passwords."""
    passwords_to_try = [
        "postgres",
        "password", 
        "",  # empty password
        "admin",
        "root",
        "123456"
    ]
    
    print("🔍 Testing PostgreSQL Connection...")
    print("=" * 50)
    
    for password in passwords_to_try:
        try:
            print(f"Trying password: '{password}'")
            
            conn = await asyncpg.connect(
                host="localhost",
                port=5433,
                user="postgres",
                password=password,
                database="postgres"
            )
            
            # Test the connection
            result = await conn.fetchval("SELECT version()")
            await conn.close()
            
            print(f"✅ SUCCESS! Password '{password}' works!")
            print(f"PostgreSQL version: {result}")
            
            # Test if the ircc_ai_platform database exists
            try:
                conn = await asyncpg.connect(
                    host="localhost",
                    port=5433,
                    user="postgres",
                    password=password,
                    database="ircc_ai_platform"
                )
                await conn.close()
                print(f"✅ Database 'ircc_ai_platform' exists and accessible")
            except Exception as e:
                print(f"⚠️  Database 'ircc_ai_platform' not accessible: {str(e)}")
            
            return password
            
        except Exception as e:
            print(f"❌ Failed with password '{password}': {str(e)}")
            continue
    
    print("\n❌ No working password found!")
    return None

async def main():
    """Main function."""
    working_password = await test_postgres_passwords()
    
    if working_password is not None:
        print(f"\n🎉 Found working password: '{working_password}'")
        print(f"\n📝 Update your .env file:")
        print(f"POSTGRES_PASSWORD={working_password}")
    else:
        print("\n❌ Could not connect to PostgreSQL")
        print("Please check if PostgreSQL is running and accessible")

if __name__ == "__main__":
    asyncio.run(main())
