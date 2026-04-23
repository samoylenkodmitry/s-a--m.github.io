---
layout: leetcode-entry
title: "3335. Total Characters in String After Transformations I"
permalink: "/leetcode/problem/2025-05-13-3335-total-characters-in-string-after-transformations-i/"
leetcode_ui: true
entry_slug: "2025-05-13-3335-total-characters-in-string-after-transformations-i"
---

[3335. Total Characters in String After Transformations I](https://leetcode.com/problems/total-characters-in-string-after-transformations-i/description) medium
[blog post](https://leetcode.com/problems/total-characters-in-string-after-transformations-i/solutions/6739800/kotlin-rust-by-samoylenkodmitry-poq9/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13052025-3335-total-characters-in?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/aZQd7flJlAE)
![1.webp](/assets/leetcode_daily_images/95c6cd43.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/987

#### Problem TLDR

String length after t shifts c=c+1,`z`=`ab` #medium #dp

#### Intuition

My chain-of-thougths:

```j

    // a + 25 = z    len = 1
    // a + 26 = z + 1 = a + b    len = 2
    // b + 24 = z
    // b + 25 = z + 1 = a + b    len = 2

    // ab + 25 = abz
    // ab + 26 = bcab

    // each char grows at its own rate
    // naive dp - TLE

```

Each char has it's own growth law that can be cached by remaining time.

Another intuition is from the hint: just simulate the process for chars frequencies.

#### Approach

* speed up by quick-jumping to `z`
* for simultaion we can jump to `z` too, full 0..26 circle would produce a new char for each existing char

#### Complexity

- Time complexity:
$$O(t + s)$$

- Space complexity:
$$O(t)$$ for dp, O(1) for the simulation

#### Code

```kotlin

// 420ms https://leetcode.com/problems/total-characters-in-string-after-transformations-i/submissions/1632533567
    fun lengthAfterTransformations(s: String, t: Int): Int {
        val M = 1000000007; val dp = HashMap<Pair<Char, Int>, Int>()
        fun dfs(c: Char, t: Int): Int = if (t == 0) 1 else dp.getOrPut(c to t) {
            if (c == 'z') (dfs('a', t - 1) + dfs('b', t - 1)) % M
            else if (t >= 'z' - c) dfs('z', t - ('z' - c)) else dfs(c + 1, t - 1)
        }
        var res = 0; for (c in s) res = (res + dfs(c, t)) % M
        return res
    }

```
```kotlin

// 57ms
    fun lengthAfterTransformations(s: String, t: Int): Int {
        val M = 1000000007; val dp = Array(26) { IntArray(t + 1) { -1 }}
        fun dfs(c: Int, t: Int): Int = if (t == 0) 1 else
            if (dp[c][t] >= 0) dp[c][t] else
            (if (c == 25) (dfs(0, t - 1) + dfs(1, t - 1)) % M
            else if (t >= 25 - c) dfs(25, t - (25 - c)) else dfs(c + 1, t - 1))
            .also { dp[c][t] = it }
        var res = 0; for (c in s) res = (res + dfs(c - 'a', t)) % M
        return res
    }

```
```kotlin

// 28ms
    fun lengthAfterTransformations(s: String, t: Int): Int {
        var f = IntArray(26); for (c in s) ++f[c - 'a']
        var f2 = IntArray(26); val M = 1000000007; var res = 0
        for (i in 1..t) {
            f2[0] = f[25]; for (c in 0..24) f2[c + 1] = f[c]
            f2[1] = (f2[1] + f[25]) % M
            f = f2.also { f2 = f }
        }
        for (c in f) res = (res + c) % M
        return res
    }

```
```kotlin

// 11ms https://leetcode.com/problems/total-characters-in-string-after-transformations-i/submissions/1632555700
    fun lengthAfterTransformations(s: String, t: Int): Int {
        var f = IntArray(26); for (c in s) ++f[c - 'a']
        val M = 1000000007; var res = 0
        for (i in 1..t / 26) for (c in 0..25) {
            val a = (26 - c) % 26; f[a] = (f[a] + f[25 - c]) % M
        }
        for (c in 0..<t % 26) {
            val a = (26 - c) % 26; f[a] = (f[a] + f[25 - c]) % M
        }
        for (c in f) res = (res + c) % M
        return res
    }

```
```rust

// 0ms https://leetcode.com/problems/total-characters-in-string-after-transformations-i/submissions/1632553112
    pub fn length_after_transformations(s: String, t: i32) -> i32 {
        let (mut f, M) = ([0; 26], 1000000007);
        for b in s.bytes() { f[(b - b'a') as usize] += 1 }
        for i in 0..t / 26 { for c in 0..26 {
                let a = (26 - c) % 26; f[a] = (f[a] + f[25 - c]) % M
        }}
        for c in 0..t as usize % 26 {
            let a = (26 - c) % 26; f[a] = (f[a] + f[25 - c]) % M
        }
        f.iter().fold(0, |r, c| (r + c) % M)
    }

```
```c++

// 6ms https://leetcode.com/problems/total-characters-in-string-after-transformations-i/submissions/1632563226
    int lengthAfterTransformations(string s, int t) {
        int f[26]={}, r = 0, M = 1e9+7; for (char c: s) ++f[c - 'a'];
        for (int i = 0; i < t / 26; ++i) for (int c = 0; c < 26; ++c) {
            int a = (26 - c) % 26; f[a] = (f[a] + f[25 - c]) % M;
        }
        for (int c = 0; c < t % 26; ++c) {
            int a = (26 - c) % 26; f[a] = (f[a] + f[25 - c]) % M;
        }
        for (int c: f) r = (r + c) % M; return r;
    }

```

