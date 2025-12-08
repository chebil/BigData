# HDFS Architecture and Design

## Learning Objectives

- Understand HDFS architecture and components
- Explain block storage and replication
- Work with HDFS command-line interface
- Implement fault tolerance mechanisms
- Optimize HDFS for Big Data workloads

## Introduction

The Hadoop Distributed File System (HDFS) is designed to store massive datasets reliably across clusters of commodity hardware. Inspired by the Google File System (GFS), HDFS provides the storage foundation for the entire Hadoop ecosystem.

## Design Principles

### 1. Hardware Failure is the Norm

**Assumption**: Failures are inevitable in large clusters

**Solution**: Automatic fault detection and recovery through replication

### 2. Streaming Data Access

**Optimization**: Sequential reads over random access

**Use Case**: Batch processing of entire datasets

### 3. Large Datasets

**Scale**: Files ranging from gigabytes to terabytes

**Design**: Optimized for high aggregate bandwidth

### 4. Simple Coherency Model

**Write-Once-Read-Many**: Files are written once and read multiple times

**Benefit**: Simplifies data consistency and enables high throughput

### 5. Moving Computation is Cheaper Than Moving Data

**Principle**: Bring computation to data, not vice versa

**Result**: Minimizes network congestion and increases throughput

## HDFS Architecture

### Master-Worker Pattern

```
HDFS Cluster
│
├── NameNode (Master)
│   ├── Manages filesystem namespace
│   ├── Maintains metadata
│   ├── Tracks block locations
│   └── Coordinates DataNode operations
│
├── DataNodes (Workers)
│   ├── Store actual data blocks
│   ├── Serve read/write requests
│   ├── Report to NameNode (heartbeats)
│   └── Perform block replication
│
└── Secondary NameNode
    └── Assists with checkpoint creation
```

### NameNode: The Master

**Responsibilities**:

1. **Namespace Management**
   - Directory structure
   - File permissions and ownership
   - File-to-block mappings

2. **Metadata Storage**
   - FsImage: Filesystem snapshot
   - EditLog: Transaction log

3. **Block Management**
   - Track block locations across DataNodes
   - Coordinate replication
   - Handle block reports

4. **Client Operations**
   - Open, close, rename files
   - List directories

**Critical Point**: NameNode is a **Single Point of Failure (SPOF)**

**Mitigation**:
- Regular backups of metadata
- HDFS High Availability (HA) with standby NameNode
- NFS-mounted storage for shared edits

### DataNodes: The Workers

**Responsibilities**:

1. **Block Storage**
   - Store blocks on local disk
   - Serve block read/write requests

2. **Heartbeat Protocol**
   - Send periodic heartbeats to NameNode (default: 3 seconds)
   - Report storage capacity and health

3. **Block Reports**
   - Send complete block inventory to NameNode
   - Typically every 6 hours

4. **Block Operations**
   - Create, delete, replicate blocks
   - Perform checksums for data integrity

### Secondary NameNode

**Purpose**: Periodic checkpoint creation (NOT a backup!)

**Process**:

1. Download FsImage and EditLog from NameNode
2. Merge them into new FsImage
3. Upload new FsImage to NameNode
4. NameNode replaces old FsImage and truncates EditLog

**Benefit**: Prevents EditLog from growing too large

## Block Storage

### Block Size

**Default**: 128 MB (Hadoop 2.x and later)

**Why Large Blocks?**

1. **Minimize Metadata**: Fewer blocks = less metadata in NameNode
2. **Reduce Seek Time**: Sequential reads dominate seek overhead
3. **Network Efficiency**: Fewer network connections per file

**Example**:
```
File Size: 1 GB
Block Size: 128 MB
Number of Blocks: 8 blocks

Block 1: 128 MB
Block 2: 128 MB
...
Block 8: 128 MB
```

**Note**: If file is smaller than block size, it only uses needed space

```python
# Conceptual example
file_size = 50 * 1024 * 1024  # 50 MB
block_size = 128 * 1024 * 1024  # 128 MB

# Only uses 50 MB on disk, not 128 MB!
disk_usage = min(file_size, block_size)
print(f"Disk usage: {disk_usage / (1024**2)} MB")
```

### Replication

**Default Replication Factor**: 3

**Benefits**:
1. **Fault Tolerance**: Data survives node failures
2. **Load Balancing**: Multiple nodes can serve same block
3. **Data Locality**: Computation can run on any replica

**Replication Pipeline**:

```
Client writes Block 1
    |
    v
DataNode A (stores + forwards)
    |
    v
DataNode B (stores + forwards)
    |
    v
DataNode C (stores + acknowledges)
    |
    v
Acknowledgment flows back to Client
```

### Rack Awareness

**Strategy**: Distribute replicas across racks for reliability

**Default Placement Policy**:
- **Replica 1**: Same node as writer (if writer is DataNode)
- **Replica 2**: Different rack from Replica 1
- **Replica 3**: Same rack as Replica 2, different node

**Benefits**:
- Survives rack-level failures
- Optimizes network bandwidth (2 replicas in same rack)

```
Rack 1              Rack 2
┌─────────┐        ┌─────────┐
│ Node A  │        │ Node D  │
│ Block 1 │        │ Block 1 │  <- Replica 2
└─────────┘        └─────────┘
                   ┌─────────┐
┌─────────┐        │ Node E  │
│ Node B  │        │ Block 1 │  <- Replica 3
└─────────┘        └─────────┘

┌─────────┐
│ Node C  │
└─────────┘
```

## HDFS Read Operation

**Step-by-Step Process**:

```python
# Conceptual flow
def hdfs_read(filename):
    # 1. Client contacts NameNode
    block_locations = namenode.get_block_locations(filename)
    
    # 2. NameNode returns block locations
    # [(block_id, [datanode1, datanode2, datanode3]), ...]
    
    # 3. Client reads from closest DataNode
    data = []
    for block_id, datanodes in block_locations:
        # Choose closest DataNode (network topology)
        datanode = choose_closest(datanodes)
        
        # 4. Read block directly from DataNode
        block_data = datanode.read_block(block_id)
        
        # 5. Verify checksum
        if not verify_checksum(block_data):
            # Try different replica
            datanode = choose_next_closest(datanodes)
            block_data = datanode.read_block(block_id)
        
        data.append(block_data)
    
    # 6. Concatenate blocks
    return b''.join(data)
```

**Key Points**:
- NameNode only provides metadata
- Actual data flows directly from DataNode to Client
- Network topology optimizes DataNode selection

## HDFS Write Operation

**Step-by-Step Process**:

```python
# Conceptual flow
def hdfs_write(filename, data):
    # 1. Client asks NameNode to create file
    namenode.create_file(filename)
    
    # 2. Split data into blocks
    blocks = split_into_blocks(data, block_size=128*1024*1024)
    
    for block_data in blocks:
        # 3. Request DataNodes for block
        datanodes = namenode.allocate_block(filename, replication=3)
        
        # 4. Write to pipeline
        # Client -> DN1 -> DN2 -> DN3
        write_pipeline(block_data, datanodes)
        
        # 5. Acknowledgment propagates back
        # DN3 -> DN2 -> DN1 -> Client
    
    # 6. Close file
    namenode.close_file(filename)
```

**Pipeline Writing**:

```
Client              DataNode 1        DataNode 2        DataNode 3
  |                     |                 |                 |
  | --- Block data ---> |                 |                 |
  |                     | -- Forward ---> |                 |
  |                     |                 | -- Forward ---> |
  |                     |                 |                 |
  |                     | <-- ACK --------|                 |
  |                     |                 | <-- ACK --------|  
  | <-- ACK -----------|                 |                 |
```

## HDFS Command Line Interface

### Basic Operations

```bash
# List files
hdfs dfs -ls /user/data
hdfs dfs -ls -h /user/data  # Human-readable sizes
hdfs dfs -ls -R /user/data  # Recursive

# Create directory
hdfs dfs -mkdir /user/data/processed
hdfs dfs -mkdir -p /user/data/year=2024/month=01  # Create parent dirs

# Upload files
hdfs dfs -put local_file.txt /user/data/
hdfs dfs -put -f local_file.txt /user/data/  # Force overwrite
hdfs dfs -copyFromLocal local_file.txt /user/data/  # Same as put

# Download files
hdfs dfs -get /user/data/file.txt .
hdfs dfs -copyToLocal /user/data/file.txt .

# Copy within HDFS
hdfs dfs -cp /user/data/file1.txt /user/backup/

# Move/rename
hdfs dfs -mv /user/data/old_name.txt /user/data/new_name.txt

# Delete
hdfs dfs -rm /user/data/file.txt
hdfs dfs -rm -r /user/data/old_directory  # Recursive delete
hdfs dfs -rm -r -skipTrash /user/data/temp  # Bypass trash

# View file content
hdfs dfs -cat /user/data/file.txt
hdfs dfs -tail /user/data/logfile.txt  # Last 1 KB
hdfs dfs -head /user/data/file.txt  # First 1 KB

# Check disk usage
hdfs dfs -du /user/data
hdfs dfs -du -h /user/data  # Human-readable
hdfs dfs -du -s /user/data  # Summary

# Change permissions
hdfs dfs -chmod 755 /user/data/file.txt
hdfs dfs -chown user:group /user/data/file.txt

# Get file statistics
hdfs dfs -stat "%b %n %r" /user/data/file.txt
# %b: File size, %n: Filename, %r: Replication factor
```

### Advanced Operations

```bash
# Check file system
hdfs fsck /user/data -files -blocks -locations

# Get block information
hdfs fsck /user/data/large_file.txt -files -blocks

# Balance cluster
hdfs balancer -threshold 10  # Balance if node usage differs by >10%

# Safe mode (maintenance)
hdfs dfsadmin -safemode enter
hdfs dfsadmin -safemode leave
hdfs dfsadmin -safemode get

# Cluster report
hdfs dfsadmin -report

# DataNode information
hdfs dfsadmin -printTopology

# Change replication factor
hdfs dfs -setrep -w 3 /user/data/important.txt
hdfs dfs -setrep -R -w 2 /user/data/archive/  # Recursive
```

## Python HDFS Interface

### Using hdfs3 Library

```python
from hdfs3 import HDFileSystem

# Connect to HDFS
hdfs = HDFileSystem(host='namenode', port=8020)

# List files
files = hdfs.ls('/user/data')
print(files)

# Read file
with hdfs.open('/user/data/file.txt', 'rb') as f:
    content = f.read()
    print(content.decode('utf-8'))

# Write file
data = b"Hello HDFS from Python!"
with hdfs.open('/user/data/new_file.txt', 'wb') as f:
    f.write(data)

# File info
info = hdfs.info('/user/data/file.txt')
print(f"Size: {info['size']} bytes")
print(f"Replication: {info['replication']}")

# Delete
hdfs.rm('/user/data/temp_file.txt')

# Check if exists
if hdfs.exists('/user/data/file.txt'):
    print("File exists")
```

### Using PyArrow with HDFS

```python
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import fs

# Connect
hdfs_fs = fs.HadoopFileSystem(host='namenode', port=8020)

# Read Parquet from HDFS
table = pq.read_table('hdfs://namenode:8020/user/data/data.parquet',
                      filesystem=hdfs_fs)
df = table.to_pandas()

# Write Parquet to HDFS
pq.write_table(table, 
               'hdfs://namenode:8020/user/data/output.parquet',
               filesystem=hdfs_fs)
```

## HDFS High Availability (HA)

### Active-Standby Configuration

```
Shared Storage (QJM or NFS)
        |
        ├── Active NameNode
        │   └── Serves all client requests
        │
        └── Standby NameNode
            └── Syncs metadata, ready for failover

ZooKeeper Cluster
└── Automatic failover coordination
```

**Components**:

1. **Active NameNode**: Handles all operations
2. **Standby NameNode**: Maintains synchronized state
3. **Shared Storage**: Stores edit logs (typically Quorum Journal Manager)
4. **ZooKeeper**: Coordinates failover
5. **ZKFailoverController (ZKFC)**: Monitors NameNode health

**Failover Process**:

```python
# Automatic failover
def monitor_namenode():
    while True:
        if not active_namenode.is_healthy():
            # 1. ZKFC detects failure
            zookeeper.fence_active_namenode()
            
            # 2. Promote standby
            standby_namenode.transition_to_active()
            
            # 3. Update ZooKeeper
            zookeeper.set_active(standby_namenode)
            
            print("Failover complete")
            break
```

## Federation

**Problem**: Single NameNode memory limits scalability

**Solution**: Multiple NameNodes, each managing a portion of namespace

```
NameNode 1          NameNode 2          NameNode 3
    |                   |                   |
/user/data         /user/logs          /user/archive
    |                   |                   |
Shared DataNode Pool (all DataNodes store blocks for all NameNodes)
```

**Benefits**:
- Horizontal scalability
- Namespace isolation
- Better performance

## Data Integrity

### Checksums

**Process**:

1. **Write**: DataNode computes checksum for each block
2. **Store**: Checksums stored alongside data
3. **Read**: Client verifies checksum
4. **Mismatch**: Try different replica, report to NameNode

```python
import hashlib

def compute_checksum(data):
    """Compute CRC32 checksum"""
    return hashlib.md5(data).hexdigest()

def verify_block(block_data, expected_checksum):
    """Verify data integrity"""
    actual_checksum = compute_checksum(block_data)
    return actual_checksum == expected_checksum
```

## Performance Tuning

### Optimal Block Size

```python
def calculate_optimal_block_size(file_size, num_mappers):
    """
    Calculate optimal HDFS block size
    
    Rule of thumb:
    - Each mapper processes 1 block
    - 10-100 mappers per node
    """
    optimal_size = file_size / num_mappers
    
    # Round to standard sizes
    if optimal_size < 64 * 1024**2:
        return 64 * 1024**2  # 64 MB
    elif optimal_size < 128 * 1024**2:
        return 128 * 1024**2  # 128 MB
    elif optimal_size < 256 * 1024**2:
        return 256 * 1024**2  # 256 MB
    else:
        return 512 * 1024**2  # 512 MB

# Example
file_size = 10 * 1024**3  # 10 GB
num_nodes = 10
mappers_per_node = 10
total_mappers = num_nodes * mappers_per_node

optimal = calculate_optimal_block_size(file_size, total_mappers)
print(f"Optimal block size: {optimal / 1024**2} MB")
```

### Replication Factor

**Guidelines**:
- **Critical data**: Replication = 5+
- **Standard data**: Replication = 3 (default)
- **Temporary data**: Replication = 2
- **Archival (rarely accessed)**: Replication = 2

## Key Takeaways

:::{admonition} Summary
:class: note

1. **HDFS uses master-worker architecture** with NameNode and DataNodes
2. **Files are split into large blocks** (128 MB default) for efficient processing
3. **Triple replication** provides fault tolerance and data locality
4. **Rack awareness** ensures reliability across hardware failures
5. **NameNode stores metadata**; actual data flows directly between clients and DataNodes
6. **High Availability** eliminates NameNode as single point of failure
7. **HDFS is optimized for large files** and sequential access patterns
8. **Data integrity** ensured through checksums and replication
:::

## Practical Exercises

### Exercise 1: HDFS Commands

Practice basic HDFS operations:

```bash
# 1. Create directory structure
hdfs dfs -mkdir -p /user/yourname/input
hdfs dfs -mkdir -p /user/yourname/output

# 2. Upload sample file
echo "Hello HDFS" > sample.txt
hdfs dfs -put sample.txt /user/yourname/input/

# 3. Check replication and block info
hdfs fsck /user/yourname/input/sample.txt -files -blocks -locations

# 4. Change replication factor
hdfs dfs -setrep 2 /user/yourname/input/sample.txt

# 5. Verify change
hdfs dfs -stat "%r" /user/yourname/input/sample.txt
```

### Exercise 2: Calculate Storage Requirements

```python
def calculate_hdfs_storage(file_size_gb, num_files, replication_factor=3):
    """
    Calculate total HDFS storage requirement
    """
    raw_data = file_size_gb * num_files  # GB
    with_replication = raw_data * replication_factor
    
    # Add 20% overhead for metadata, temporary files
    total_storage = with_replication * 1.2
    
    return {
        'raw_data_gb': raw_data,
        'with_replication_gb': with_replication,
        'total_required_gb': total_storage
    }

# Example
result = calculate_hdfs_storage(file_size_gb=100, num_files=1000, replication_factor=3)
print(result)
```

## Further Reading

- [HDFS Architecture Guide](https://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html)
- Shvachko, K. et al. (2010). "The Hadoop Distributed File System"
- White, T. (2015). "Hadoop: The Definitive Guide", Chapter 3
