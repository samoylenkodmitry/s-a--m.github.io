---
layout: leetcode-entry
title: "756. Pyramid Transition Matrix"
permalink: "/leetcode/problem/2025-12-29-756-pyramid-transition-matrix/"
leetcode_ui: true
entry_slug: "2025-12-29-756-pyramid-transition-matrix"
---

[756. Pyramid Transition Matrix](https://leetcode.com/problems/pyramid-transition-matrix/description/) medium
[blog post](https://leetcode.com/problems/pyramid-transition-matrix/solutions/7448322/kotlin-rust-by-samoylenkodmitry-e1kz/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29122025-756-pyramid-transition-matrix?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/82UTGBKUFGA)

![a2c53c7b-0476-42f6-8094-e3142145e06c (1).webp](/assets/leetcode_daily_images/7921098c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1219

#### Problem TLDR

Can build a pyramid from transitions? #medium #backtracking

#### Intuition

DFS for each row.
DFS with backtracking for each transition character in the row.

#### Approach

* speedup with arena allocation
* speedup with Int keys: A<<16|B
* speedup with transitions matrix bitset: M[A][B] = bitmask
* speedup with HashSet of bad keys

#### Complexity

- Time complexity:
$$O(b^a)$$

- Space complexity:
$$O(b^2)$$

#### Code

```kotlin
// 15ms
    fun pyramidTransition(b: String, a: List<String>): Boolean {
        val m = IntArray(26 * 26); var j = b.length; val bad = HashSet<Long>()
        val r = IntArray(j * (j + 1) / 2) { b[it % j] - 'A' }
        for (s in a) { val x = s[0]-'A'; val y = s[1]-'A'; m[x*26+y] = m[x*26+y] or (1 shl (s[2]-'A'))}
        fun dfs(st: Int, sz: Int, key: Long): Boolean {
            if (j == r.size) return true; if (key in bad) return false
            if (j == st + sz + 1 + sz) return dfs(j - sz, sz - 1, key)
            var mask = m[r[j - sz - 1] * 26 + r[j - sz]]
            while (mask > 0) {
                val c = mask.countTrailingZeroBits(); mask = mask and (mask - 1); r[j++] = c
                if (dfs(st, sz, key or (c.toLong() shl ((j-st-sz)*5)))) return true; --j
            }
            bad.add(key); return false
        }
        return dfs(0, j - 1, 0)
    }
```
```kotlin
// 400ms
    fun pyramidTransition(b: String, a: List<String>): Boolean {
        val a = a.groupBy({ it.take(2) },{it[2]})
        fun dfs(b: List<Char>): Boolean {
            val r = ArrayList<Char>()
            fun dfs2(i: Int): Boolean = if (i == b.size) dfs(r) else a["${b[i-1]}${b[i]}"]
                ?.any { r += it; val n = dfs2(i+1); r.removeLast(); n } == true
            return b.size == 1 || dfs2(1)
        }
        return dfs(b.toList())
    }
```
```rust
// 0ms
    pub fn pyramid_transition(b: String, a: Vec<String>) -> bool {
        let (mut m, mut j, mut bad) = ([0u32; 26 * 26], b.len(), HashSet::<u64>::new());
        for s in a { let t = s.as_bytes(); m[(t[0]-b'A')as usize*26 + (t[1]-b'A')as usize] |= 1u32<<(t[2]-b'A')}
        let mut r = vec![0u8; j*(j+1)/2]; for (i,&c) in b.as_bytes().iter().enumerate() {r[i] = c - b'A'}
        fn dfs(m: &[u32; 26 * 26], r: &mut [u8], j: &mut usize, st: usize, sz: usize, key: u64, bad: &mut HashSet<u64>) -> bool {
            if *j == r.len() { return true }; if *j == st+sz+1+sz { return dfs(m, r, j, *j-sz, sz-1, key, bad)}
            if bad.contains(&key) { return false }
            let mut mask = m[r[*j - sz - 1] as usize * 26 + r[*j - sz] as usize];
            while mask != 0 {
                let c = mask.trailing_zeros() as u8; mask &= mask - 1; r[*j] = c; *j += 1;
                if dfs(m, r, j, st, sz, key | ((c as u64) << ((*j-st-sz-2)*5)), bad) { return true }; *j -= 1;
            }
            bad.insert(key); false
        }
        dfs(&m, &mut r, &mut j, 0, b.len() - 1, 0, &mut bad)
    }
```

