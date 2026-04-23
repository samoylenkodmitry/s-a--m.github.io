---
layout: leetcode-entry
title: "1061. Lexicographically Smallest Equivalent String"
permalink: "/leetcode/problem/2025-06-05-1061-lexicographically-smallest-equivalent-string/"
leetcode_ui: true
entry_slug: "2025-06-05-1061-lexicographically-smallest-equivalent-string"
---

[1061. Lexicographically Smallest Equivalent String](https://leetcode.com/problems/lexicographically-smallest-equivalent-string/description/) medium
[blog post](https://leetcode.com/problems/lexicographically-smallest-equivalent-string/solutions/6812801/kotlin-rust-by-samoylenkodmitry-hinl/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05062025-1061-lexicographically-smallest?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/NdpQusEXyjc)
![1.webp](/assets/leetcode_daily_images/b1053864.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1010

#### Problem TLDR

Map by smallest in group from associate s1 with s2 #medium

#### Intuition

We can do DFS or use a Union-Find to track groups.

#### Approach

* we can find minimum in-place by always picking it as root
* no reason to optimize (compression, ranking) Union-Find of just 26 elements

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$, for the result

#### Code

```kotlin

// 30ms
    fun smallestEquivalentString(s1: String, s2: String, bs: String): String {
        val u = HashMap(('a'..'z').associateBy { it })
        fun f(x: Char): Char = if (x == u[x]) x else f(u[x]!!)
        for ((a, b) in s1.zip(s2)) if (f(a) < f(b)) u[f(b)] = f(a) else u[f(a)] = f(b)
        return bs.map(::f).joinToString("")
    }

```
```kotlin

// 2ms
    fun smallestEquivalentString(s1: String, s2: String, bs: String): String {
        val u = IntArray(26) { it }; val a = bs.toCharArray()
        fun f(a: Int): Int { var x = a; while (x != u[x]) x = u[x]; u[a] = x; return x }
        for (i in s1.indices) {
            val a = f(s1[i] - 'a'); val b = f(s2[i] - 'a')
            if (a < b) u[b] = a else u[a] = b
        }
        for (i in a.indices) a[i] = 'a' + f(a[i] - 'a'); return String(a)
    }

```
```rust

// 0ms
    pub fn smallest_equivalent_string(s1: String, s2: String, bs: String) -> String {
        let mut u: Vec<_> = (0..26).collect();
        fn f(x: u8, u: &mut Vec<usize>) -> usize { let x = x as usize; while u[x] != u[u[x]] { u[x] = u[u[x]] }; u[x] }
        for (a, b) in s1.bytes().zip(s2.bytes()) {
            let (a, b) = (f(a - b'a', &mut u), f(b - b'a', &mut u)); if a < b { u[b] = a } else { u[a] = b }
        }
        bs.bytes().map(|b| (b'a' + f(b - b'a', &mut u) as u8) as char).collect()
    }

```
```c++

// 0ms
    string smallestEquivalentString(string s1, string s2, string bs) {
        int u[26] = {}; iota(u, u + 26, 0);
        auto f = [&](int x) { while (x != u[x]) x = u[x]; return x; };
        for (int i = 0; i < size(s1); ++i) {
            int a = f(s1[i] - 'a'), b = f(s2[i] - 'a');
            u[max(a, b)] = min(a, b);
        }
        for (char& c: bs) c = 'a' + f(c - 'a'); return bs;
    }

```

