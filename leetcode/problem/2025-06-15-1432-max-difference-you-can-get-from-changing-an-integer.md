---
layout: leetcode-entry
title: "1432. Max Difference You Can Get From Changing an Integer"
permalink: "/leetcode/problem/2025-06-15-1432-max-difference-you-can-get-from-changing-an-integer/"
leetcode_ui: true
entry_slug: "2025-06-15-1432-max-difference-you-can-get-from-changing-an-integer"
---

[1432. Max Difference You Can Get From Changing an Integer](https://leetcode.com/problems/max-difference-you-can-get-from-changing-an-integer/description/) medium
[blog post](https://leetcode.com/problems/max-difference-you-can-get-from-changing-an-integer/solutions/6845130/kotlin-rust-by-samoylenkodmitry-7kre/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15062025-1432-max-difference-you?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/eLw0XB49cJA)
![1.webp](/assets/leetcode_daily_images/78d4845d.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1020

#### Problem TLDR

Max - min, replacing any digit, no leading zeros #medium

#### Intuition

The brute force is fast enough.
More optimized:
* max is first non-nine replaced by 9
* min is if first is non-one, replace by 1
* else first non-zero replaced by 0 BUT

#### Approach

* also can be solved without convertion to strings

#### Complexity

- Time complexity:
$$O(1)$$, 10 digits, 10x10 runs, total is 1000 operations; 10 ops for optimized

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 12ms
    fun maxDiff(n: Int): Int {
        var max = n; var min = n
        for (a in "$n") for (b in "0123456789") {
            val x = "$n".replace(a, b)
            if (x == "0" || x[0] == '0') continue
            max = max(max, x.toInt())
            min = min(min, x.toInt())
        }
        return max - min
    }

```
```kotlin

// 8ms
    fun maxDiff(n: Int) =
        "$n".replace("$n".find { it != '9' } ?: '.', '9').toInt() -
        if ("$n"[0] > '1') "$n".replace("$n"[0], '1').toInt()
        else "$n".replace("$n".find { it != "$n"[0] && it > '0' } ?: '.', '0').toInt()

```
```rust

// 0ms
    pub fn max_diff(n: i32) -> i32 {
        let (s, mut a, mut b) = (n.to_string(), n, n);
        for c in s.chars() {
            for d in '0'..='9' {
                let x: String = s.chars().map(|x| if x == c { d } else { x }).collect();
                if x == "0" || x.starts_with("0") { continue }
                let x = x.parse().unwrap();
                a = a.max(x); b = b.min(x)
            }
        } a - b
    }

```
```c++

// 0ms
    int maxDiff(int n) {
        string s = to_string(n);
        int a = n, b = n;
        for (char c: s) for (char d = '0'; d <= '9'; ++d) {
            string t = s;
            for (char& x: t) if (x == c) x = d;
            if (t == "0" || t[0] == '0') continue;
            int x = stoi(t); a = max(a, x); b = min(b, x);
        } return a - b;
    }

```

