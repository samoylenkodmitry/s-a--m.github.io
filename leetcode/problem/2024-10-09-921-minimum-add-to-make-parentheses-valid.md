---
layout: leetcode-entry
title: "921. Minimum Add to Make Parentheses Valid"
permalink: "/leetcode/problem/2024-10-09-921-minimum-add-to-make-parentheses-valid/"
leetcode_ui: true
entry_slug: "2024-10-09-921-minimum-add-to-make-parentheses-valid"
---

[921. Minimum Add to Make Parentheses Valid](https://leetcode.com/problems/minimum-add-to-make-parentheses-valid/description/) medium
[blog post](https://leetcode.com/problems/minimum-add-to-make-parentheses-valid/solutions/5890400/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09102024-921-minimum-add-to-make?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/vesQpdacg44)
![1.webp](/assets/leetcode_daily_images/f1b0223d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/762

#### Problem TLDR

Minimum inserts to balance brackets #medium #stack

#### Intuition

The optimal way to return the balance is to insert lazily on every unbalanced position. (Prove is out of scope)

Now, to check the balance, let's use a stack and match each open bracket with the closing. We can simplify the stack down to the `counter`.

#### Approach

* keep the balance variable `b` separate from the insertions' count variable `res`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minAddToMakeValid(s: String): Int {
        var b = 0; var res = 0
        for (c in s) if (c == '(') b++
            else if (b > 0) b-- else res++
        return res + b
    }

```
```rust

    pub fn min_add_to_make_valid(s: String) -> i32 {
        let (mut b, mut res) = (0, 0);
        for c in s.bytes() {
            if c == b'(' { b += 1 } else if b > 0 { b -= 1 }
            else { res += 1 }
        }; res + b
    }

```
```c++

    int minAddToMakeValid(string s) {
        int b = 0, res = 0;
        for (char c: s) if (c == '(') b++;
            else if (b > 0) b--; else res++;
        return res + b;
    }

```

