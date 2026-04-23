---
layout: leetcode-entry
title: "1980. Find Unique Binary String"
permalink: "/leetcode/problem/2025-02-20-1980-find-unique-binary-string/"
leetcode_ui: true
entry_slug: "2025-02-20-1980-find-unique-binary-string"
---

[1980. Find Unique Binary String](https://leetcode.com/problems/find-unique-binary-string/description/) medium
[blog post](https://leetcode.com/problems/find-unique-binary-string/solutions/6445841/kotlin-rust-by-samoylenkodmitry-k733/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20022025-1980-find-unique-binary?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/DQkbAHYP3ZY)
![1.webp](/assets/leetcode_daily_images/1c7a4909.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/901

#### Problem TLDR

Binary number not in a set #medium #backtracking

#### Intuition

Several solutions:
* brute-force: construct every number with DFS+backtracking or in a (0..2^n) loop and take first not in a set
* iterative: sort the list, iterate and take frist `i != n[i]`
* from /u/votrubac/: `n` is much less than `2^n`, so random number have a good chance of `f(n) = (1 - n / 2 ^ n)`, f(10) = 0.99, f(5) = 0.84
* from /u/votrubac/ & Cantor: `If s1, s2, ... , sn, ... is any enumeration of elements from T,[note 3] then an element s of T can be constructed that doesn't correspond to any sn in the enumeration.` meaning we can always add binary number to the set. (https://en.wikipedia.org/wiki/Cantor%27s_diagonal_argument)

#### Approach

* personally, the backtracking is the only solution I could invent on the fly

#### Complexity

- Time complexity:
$$O(2^n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun findDifferentBinaryString(n: Array<String>, s: String = ""): String? =
        if (s.length < n[0].length) findDifferentBinaryString(n, s + "0")
        ?: findDifferentBinaryString(n, s + "1") else if (s in n) null else s

```
```rust

    pub fn find_different_binary_string(mut n: Vec<String>) -> String {
        n.sort(); format!("{:0w$b}", n.iter().enumerate()
        .find(|&(i, s)| i != usize::from_str_radix(s, 2).unwrap())
        .unwrap_or((n.len(), &"".into())).0, w = n.len())
    }

```
```c++

    string findDifferentBinaryString(vector<string>& n) {
        for (int i = 0; i < size(n); ++i) n[0][i] = '0' + '1' - n[i][i];
        return n[0];
    }

```

