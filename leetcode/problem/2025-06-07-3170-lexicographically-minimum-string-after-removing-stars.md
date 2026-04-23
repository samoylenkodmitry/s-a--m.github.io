---
layout: leetcode-entry
title: "3170. Lexicographically Minimum String After Removing Stars"
permalink: "/leetcode/problem/2025-06-07-3170-lexicographically-minimum-string-after-removing-stars/"
leetcode_ui: true
entry_slug: "2025-06-07-3170-lexicographically-minimum-string-after-removing-stars"
---

[3170. Lexicographically Minimum String After Removing Stars](https://leetcode.com/problems/lexicographically-minimum-string-after-removing-stars/description) medium
[blog post](https://leetcode.com/problems/lexicographically-minimum-string-after-removing-stars/solutions/6819367/kotlin-rust-by-samoylenkodmitry-64zr/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07062025-3170-lexicographically-minimum?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/9DpHn1XL-hs)
![1.webp](/assets/leetcode_daily_images/e8b61fb0.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1012

#### Problem TLDR

Smallest string by remove min to the left of * #medium

#### Intuition

```j
    // aab*b*c*c*
    //  . *         should go left to right
    // .    *
    //     .  *
    //   .      *
```

We have to remove rightmost of the smallest.

#### Approach

* track indices
* can use a heap

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 381ms
    fun clearStars(s: String): String {
        val a = s.toCharArray()
        val q = PriorityQueue<Int>(compareBy({ s[it] }, { -it }))
        for (i in s.indices) if (a[i] == '*') a[q.poll()] = '*' else q += i
        return a.filter { it != '*' }.joinToString("")
    }

```
```kotlin

// 60ms https://leetcode.com/problems/lexicographically-minimum-string-after-removing-stars/submissions/1656351997
    fun clearStars(s: String): String {
        val a = s.toCharArray(); var j = 0; var k = 26
        val f = Array(26) { ArrayList<Int>() }
        for (i in s.indices) if (a[i] == '*') {
            a[f[k].removeLast()] = '*'
            while (k < 26 && f[k].size == 0) k++
        } else { f[a[i] - 'a'] += i; k = min(k, a[i] - 'a') }
        for (i in a.indices) if (a[i] != '*') a[j++] = a[i]
        return String(a, 0, j)
    }

```

```rust

// 19ms
    pub fn clear_stars(mut s: String) -> String {
        let (mut b, mut k, mut f) = (unsafe { s.as_bytes_mut() }, 26, vec![vec![]; 26]);
        for i in 0..b.len() {
            if b[i] == b'*' {
                b[f[k].pop().unwrap()] = b'*';
                while k < 26 && f[k].len() == 0 { k += 1 }
            } else { let b = (b[i] - b'a') as usize; f[b].push(i); k = k.min(b) }}
        s.retain(|c| c != '*'); s
    }

```
```c++

// 29ms
    string clearStars(string s) {
        array<vector<int>, 26> f; int k = 26;
        for (int i = 0; i < size(s); ++i)
            if (s[i] != '*') f[s[i] - 'a'].push_back(i), k = min(k, s[i] - 'a');
            else { s[f[k].back()] = '*'; f[k].pop_back(); while (k < 26 && !size(f[k])) ++k; }
        erase(s, '*'); return s;
    }

```

