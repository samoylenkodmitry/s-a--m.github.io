---
layout: leetcode-entry
title: "1717. Maximum Score From Removing Substrings"
permalink: "/leetcode/problem/2024-07-12-1717-maximum-score-from-removing-substrings/"
leetcode_ui: true
entry_slug: "2024-07-12-1717-maximum-score-from-removing-substrings"
---

[1717. Maximum Score From Removing Substrings](https://leetcode.com/problems/maximum-score-from-removing-substrings/description/) medium
[blog post](https://leetcode.com/problems/maximum-score-from-removing-substrings/solutions/5464003/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12072024-1717-maximum-score-from?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/EPDGeeTfNeY)
![2024-07-12_08-17_1.webp](/assets/leetcode_daily_images/44cac0c1.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/667

#### Problem TLDR

Max score removing from `s`, `x` for `ab`, `y` for `ba` #medium #greedy #stack

#### Intuition

The first intuition is to remove greedily, but how exactly? Let's observe some examples:

```j

    // aba      x=1 y=2
    // a     a
    //  b    ab
    //
    // aabbab   x=1 y=2  y>x
    // a      a
    //  a     aa
    //   b    aab
    //
    // bbaabb  x>y
    // b      b
    //  b     bb
    //   a    bba
    //    a   bb
    // ...

```
We should maintain the Stack to be able to remove cases like `aabb` in one go.
We should `not` remove the first `ab` from `aba`, when the reward from `ba` is larger. So, let's do it in two passes: first remove the larger reward, then the other one.

#### Approach

* we can extract the removal function

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun maximumGain(s: String, x: Int, y: Int): Int {
        var points = 0
        val a = if (x > y) 'a' else 'b'; val b = if (a < 'b') 'b' else 'a'
        val stack = Stack<Char>().apply {
            for (c in s) if (c == b && size > 0 && peek() == a) {
                pop(); points += max(x, y)
            } else push(c)
        }
        Stack<Char>().apply {
            for (c in stack) if (c == a && size > 0 && peek() == b) {
                    pop(); points += min(x, y)
                } else push(c)
        }
        return points
    }

```
```rust

    pub fn maximum_gain(s: String, mut x: i32, mut y: i32) -> i32 {
        let (mut a, mut b) = (b'a', b'b');
        if x < y { mem::swap(&mut a, &mut b); mem::swap(&mut x, &mut y) }
        fn remove_greedy(s: &String, a: u8, b: u8) -> String {
            let mut res = vec![];
            for c in s.bytes() {
                if res.len() > 0 && *res.last().unwrap() == a && c == b {
                    res.pop();
                } else { res.push(c) }
            }
            String::from_utf8(res).unwrap()
        }
        let s1 = remove_greedy(&s, a, b); let s2 = remove_greedy(&s1, b, a);
        (s.len() - s1.len()) as i32 / 2 * x + (s1.len() - s2.len()) as i32 / 2 * y
    }

```

