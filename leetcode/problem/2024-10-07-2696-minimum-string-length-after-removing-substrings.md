---
layout: leetcode-entry
title: "2696. Minimum String Length After Removing Substrings"
permalink: "/leetcode/problem/2024-10-07-2696-minimum-string-length-after-removing-substrings/"
leetcode_ui: true
entry_slug: "2024-10-07-2696-minimum-string-length-after-removing-substrings"
---

[2696. Minimum String Length After Removing Substrings](https://leetcode.com/problems/minimum-string-length-after-removing-substrings/description/) easy
[blog post](https://leetcode.com/problems/minimum-string-length-after-removing-substrings/solutions/5880917/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07102024-2696-minimum-string-length?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Qwo3puWGmt0)
![1.webp](/assets/leetcode_daily_images/cfd5edcd.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/759

#### Problem TLDR

Remove 'AB' and 'CD' from the string #easy #stack

#### Intuition

We can do the removals in a loop until the string size changes.
However, the optimal way is to do this with a `Stack`: pop if stack top and the current char form the target to remove.

#### Approach

* Rust has a nice `match` to shorten the code

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun minLength(s: String)= Stack<Char>().run {
        for (c in s) if (size > 0 &&
            (c == 'B' && peek() == 'A' || c == 'D' && peek() == 'C'))
            pop() else push(c)
        size
    }

```
```rust

    pub fn min_length(s: String) -> i32 {
        let mut stack = vec![];
        for b in s.bytes() { match b {
            b'B' if stack.last() == Some(&b'A') => { stack.pop(); }
            b'D' if stack.last() == Some(&b'C') => { stack.pop(); }
            _ => { stack.push(b) }
        }}
        stack.len() as i32
    }

```
```c++

    int minLength(string s) {
        stack<char> st;
        for (char c: s) if (!st.empty() && (
            st.top() == 'A' && c == 'B' || st.top() == 'C' && c == 'D'
        )) st.pop(); else st.push(c);
        return st.size();
    }

```

