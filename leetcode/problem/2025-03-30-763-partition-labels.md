---
layout: leetcode-entry
title: "763. Partition Labels"
permalink: "/leetcode/problem/2025-03-30-763-partition-labels/"
leetcode_ui: true
entry_slug: "2025-03-30-763-partition-labels"
---

[763. Partition Labels](https://leetcode.com/problems/partition-labels/description) medium
[blog post](https://leetcode.com/problems/partition-labels/solutions/6595272/kotlin-rust-by-samoylenkodmitry-9n5l/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30032025-763-partition-labels?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/5GXk_BnBIsc)
![1.webp](/assets/leetcode_daily_images/d51d6175.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/943

#### Problem TLDR

Split string into separate chars sets #medium

#### Intuition

Look at the last index of the char, split when maximum of the last is current.

#### Approach

* can you do it one-pass?
* Golf: Kotlin - associate, Rust - chunked_by

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$ or O(result)

#### Code

```kotlin

    fun partitionLabels(s: String) = buildList<Int> {
        var l = -1; var r = 0; val e = s.indices.associate { s[it] to it }
        for (i in s.indices) {
            r = max(r, e[s[i]]!!); if (i == r) { add(r - l); l = i }
        }
    }

```
```kotlin

    fun partitionLabels(s: String) = buildList<Int> {
        val b = IntArray(26) { -1 }
        for ((i, c) in s.withIndex()) {
            if (b[c - 'a'] < 0) { b[c - 'a'] = size; this += 0 }
            while (b[c - 'a'] < size - 1) {
                this[size - 2] += removeLast()
                for (k in 0..25) if (b[k] == size) b[k]--
            }
            this[size - 1]++
        }
    }

```
```rust

    pub fn partition_labels(s: String) -> Vec<i32> {
        let (mut s, mut e, mut r, mut l) = (s.as_bytes(), vec![0; 26], 0, 0);
        for i in 0..s.len() { e[(s[i] - b'a') as usize] = i }
        s.chunk_by(|&c, _| { l += 1; r = r.max(e[(c - b'a') as usize]); l <= r })
        .map(|ch| ch.len() as _).collect()
    }

```
```c++

    vector<int> partitionLabels(string s) {
        int e[26]; for (int i = 0; i < size(s); ++i) e[s[i] - 'a'] = i;
        vector<int> res;
        for (int i = 0, r = 0, l = -1; i < size(s); ++i)
            if ((r = max(r, e[s[i] - 'a'])) == i) res.push_back(r - l), l = i;
        return res;
    }

```

