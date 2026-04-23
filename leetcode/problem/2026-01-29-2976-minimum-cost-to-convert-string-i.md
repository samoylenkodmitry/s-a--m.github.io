---
layout: leetcode-entry
title: "2976. Minimum Cost to Convert String I"
permalink: "/leetcode/problem/2026-01-29-2976-minimum-cost-to-convert-string-i/"
leetcode_ui: true
entry_slug: "2026-01-29-2976-minimum-cost-to-convert-string-i"
---

[2976. Minimum Cost to Convert String I](https://leetcode.com/problems/minimum-cost-to-convert-string-i/description/) medium
[blog post](https://leetcode.com/problems/minimum-cost-to-convert-string-i/solutions/7533797/kotlin-rust-by-samoylenkodmitry-3t78/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29012026-2976-minimum-cost-to-convert?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/8Oy5jS9s0nk)

![adb99ef8-8026-4882-bd51-ff5f06585d9b (2).webp](/assets/leetcode_daily_images/43f72c24.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1252

#### Problem TLDR

Min cost to convert string #medium #floyd-warshall

#### Intuition

Prepare transition matrix from any char to any char.
Path relaxation: repeat path length, repeat(path length) a,b,c ab = min(ab,ac+cb)
Floyd-Warshall: outer loop is the middle, try to update ab = min(ab, ac+cb)

```j
    // 1. find full min cost transition matrix from any char to any char
    // 2. convert source to target sumOf { m[s[i]][t[i]] }
    //
    // 1 can be done with dfs 26^2
    //
    // floyd-warshall abc    ac = min(ab+bc,ac)
    //                      wrong answer  ? (forgot about duplicates)
    //
    //
```

#### Approach

* the costs have duplicates, use min

#### Complexity

- Time complexity:
$$O(n + fw)$$

- Space complexity:
$$O(n + fw)$$

#### Code

```kotlin
// 33ms
    fun minimumCost(s: String, t: String, o: CharArray, c: CharArray, ct: IntArray): Long {
        val m = Array(26) { i -> LongArray(26) { if (i == it) 0 else 1 shl 30 }}
        for (i in o.indices) m[o[i]-'a'][c[i]-'a'] = min(1L*ct[i], m[o[i]-'a'][c[i]-'a'])
        for (c in 0..25) for (a in 0..25) for (b in 0..25) m[a][b] = min(m[a][b], m[a][c] + m[c][b])
        return s.indices.sumOf { m[s[it]-'a'][t[it]-'a'].takeIf {it < 1L shl 30}?: return -1L}
    }
```
```rust
// 6ms
    pub fn minimum_cost(s: String, t: String, o: Vec<char>, c: Vec<char>, ct: Vec<i32>) -> i64 {
        let mut m = [[1<<30;26];26]; for i in 0..26 { m[i][i] = 0 }
        for i in 0..o.len() { let (a,b) = (o[i]as usize-97, c[i]as usize-97); m[a][b]=m[a][b].min(ct[i] as i64)}
        for c in 0..26 { for a in 0..26 { for b in 0..26 { m[a][b] = m[a][b].min(m[a][c]+m[c][b])}}}
        s.bytes().zip(t.bytes()).try_fold(0, |s, (a, b)| { let c = m[(a - 97) as usize][(b - 97) as usize];
        if c >= 1 << 30 { None } else { Some(s + c) } }).unwrap_or(-1)
    }
```

