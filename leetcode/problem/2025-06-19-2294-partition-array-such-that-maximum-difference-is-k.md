---
layout: leetcode-entry
title: "2294. Partition Array Such That Maximum Difference Is K"
permalink: "/leetcode/problem/2025-06-19-2294-partition-array-such-that-maximum-difference-is-k/"
leetcode_ui: true
entry_slug: "2025-06-19-2294-partition-array-such-that-maximum-difference-is-k"
---

[2294. Partition Array Such That Maximum Difference Is K](https://leetcode.com/problems/partition-array-such-that-maximum-difference-is-k/description) medium
[blog post](https://leetcode.com/problems/partition-array-such-that-maximum-difference-is-k/solutions/6860555/kotlin-rust-by-samoylenkodmitry-gciy/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19062025-2294-partition-array-such?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/cB2lX8VwU1I)
![1.webp](/assets/leetcode_daily_images/e00b237f.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1024

#### Problem TLDR

k-narrow subsequences #medium #sort

#### Intuition

Sort, then expand each subsequence as much as possible, the tail with thank you.

#### Approach

* we can use bucket sort or a bitset

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 61ms
    fun partitionArray(n: IntArray, k: Int): Int {
        n.sort(); var l = -k - 1
        return n.count { r -> (r - l > k).also { if (it) l = r } }
    }

```
```kotlin

// 9ms
    fun partitionArray(n: IntArray, k: Int): Int {
        val f = java.util.BitSet(100001); for (x in n) f.set(x)
        var cnt = 0; var l = -k-1; var r = f.nextSetBit(0)
        while (r >= 0) {
            if (r - l > k) { l = r; ++cnt }
            r = f.nextSetBit(r + 1)
        }
        return cnt
    }

```
```kotlin

// 8ms
    fun partitionArray(n: IntArray, k: Int): Int {
        val f = IntArray(100001); for (x in n) f[x] = x + 1
        var cnt = 0; var l = -k
        for (r in f) if (r - l > k) { l = r; ++cnt }
        return cnt
    }

```
```rust

// 1ms
    pub fn partition_array(n: Vec<i32>, k: i32) -> i32 {
        let (mut f, mut l) = ([0; 100001], -k);
        for x in n { f[x as usize] = x + 1 }
        f.iter().filter(|&&r| { let w = r - l > k; if w { l = r }; w }).count() as _
    }

```
```c++

// 4ms
    int partitionArray(vector<int>& n, int k) {
        bitset<100001>f; int l = -k-1, c = 0;
        for (int x: n) f[x] = 1;
        for (int r = 0; r < 100001; ++r) if (f[r] && r - l > k) ++c, l = r;
        return c;
    }

```

