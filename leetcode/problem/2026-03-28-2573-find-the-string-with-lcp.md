---
layout: leetcode-entry
title: "2573. Find the String with LCP"
permalink: "/leetcode/problem/2026-03-28-2573-find-the-string-with-lcp/"
leetcode_ui: true
entry_slug: "2026-03-28-2573-find-the-string-with-lcp"
---

[2573. Find the String with LCP](https://open.substack.com/pub/dmitriisamoilenko/p/28032026-2573-find-the-string-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) hard

[youtube](https://youtu.be/psiEG5puckI)

![28.03.2026.webp](/assets/leetcode_daily_images/28.03.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1311

#### Problem TLDR

Build string from longest common prefixes matrix #hard

#### Intuition

```j
  // [[4,0,2,0], 0..=4, [0..][2..]=2
    //  [0,3,0,1],
    //  [2,0,2,0],
    //  [0,1,0,1]]
    // [0][0] 4=aaaa,
    // [0][1] 0= b b
    // [0][2] 2=ab
    // [0][3] 0 a!=b
    // 6 minute:
    // i have no idea
    // diagonal 4 3 2 1 is always like this
    //                  can be ignored
    // is symmetric, look only top right corner
    //
    // [[4,0,2,0],
    //  [0,3,0,1],
    //  [2,0,2,0],
    //  [0,1,0,1]]
    //
    //  020
    // abcd
    //  *   a!=b
    //   *  ab==cd
    //    * a!=d
    //   01
    //  bcd
    //   *  b!=c
    //    * b==d
    //    0
    //   cd
    //    * c!=d
    //
    // how to use this?
    // can just increment from 'a'?
    // abcd
    // aaaa
    // a!=b: abaa
    // ab==cd: abab
    // a!=d: true
    // b!=c: true
    // b==d: true
    // c!=d: true
    // will this work? let's try
    //
    // 32 minute, corner case:
    // [[4,1,1,1],
    //  [1,3,1,1],
    //  [1,1,2,1],
    //  [1,1,1,1]]
    // 38 minute, test case "abcdefghijklmnopqrstuvwxyz{"
```

n^3: check matrix symmetry and diagonal properties, build string in (i,j, count) loop, validate separately
n^2: build string filling the letters on vacant equal places, validate by dp property p[i][j]=s[i]==s[j] + p[i+1][j+1]

#### Approach

* diagonal: 4 3 2 1 always
* symmetry above vs below diagonal
* letter not overflow 'z'

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 21ms
    fun findTheString(p: Array<IntArray>): String {
        val n = p.size-1; val s = CharArray(n+1); var c = 'a'-1
        for (i in 0..n) if (s[i] < 'a') {
            if (++c > 'z') return ""
            for (j in i..n) if (p[i][j] > 0) s[j] = c
        }
        return if ((0..n).all { i -> (0..n).all { j ->
            val x = if (max(i,j) < n) p[i+1][j+1] else 0
            p[i][j] == if (s[i] == s[j]) 1 + x else 0 }}) String(s) else ""
    }
```
```rust
// 7ms
    pub fn find_the_string(p: Vec<Vec<i32>>) -> String {
        let n = p.len(); let (mut s, mut c) = (vec![0;n], 96);
        for i in 0..n { if s[i] == 0 {
            c += 1; if c > 122 { return "".into() }
            for j in 0..n { if p[i][j] > 0 { s[j] = c }}
        }}
        ((0..n).all(|i| (0..n).all(|j| p[i][j]==(s[i]==s[j])as i32 *
            (1 + if i.max(j) < n-1 { p[i+1][j+1] } else {0})
        ))).then(|| String::from_utf8(s).unwrap()).unwrap_or_default()
    }
```

