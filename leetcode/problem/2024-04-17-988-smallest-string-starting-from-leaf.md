---
layout: leetcode-entry
title: "988. Smallest String Starting From Leaf"
permalink: "/leetcode/problem/2024-04-17-988-smallest-string-starting-from-leaf/"
leetcode_ui: true
entry_slug: "2024-04-17-988-smallest-string-starting-from-leaf"
---

[988. Smallest String Starting From Leaf](https://leetcode.com/problems/smallest-string-starting-from-leaf/description/) medium
[blog post](https://leetcode.com/problems/smallest-string-starting-from-leaf/solutions/5035072/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17042024-988-smallest-string-starting?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/rDcok_WRbQo)
![2024-04-17_08-17.webp](/assets/leetcode_daily_images/a75c3efe.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/574

#### Problem TLDR

Smallest string from `leaf` to `root` in a Binary Tree #medium

#### Intuition

After trying some examples with bottom-up approach, we find out one that would not work:
![2024-04-17_08-02.webp](/assets/leetcode_daily_images/98d75c46.webp)
That means, we should use top down.

#### Approach

* We can avoid using a global variable, comparing the results.
* The `if` branching can be smaller if we add some symbol after `z` for a single-leafs.

#### Complexity

- Time complexity:
$$O(nlog^2(n))$$, we prepending to string with length of log(n) log(n) times, can be avoided with StringBuilder and reversing at the last step

- Space complexity:
$$O(log(n))$$, recursion depth

#### Code

```kotlin

    fun smallestFromLeaf(root: TreeNode?, s: String = ""): String = root?.run {
        val s = "${'a' + `val`}" + s
        if (left == null && right == null) s
        else minOf(smallestFromLeaf(left, s), smallestFromLeaf(right, s))
    } ?: "${ 'z' + 1 }"

```
```rust

    pub fn smallest_from_leaf(root: Option<Rc<RefCell<TreeNode>>>) -> String {
        fn dfs(n: &Option<Rc<RefCell<TreeNode>>>, s: String) -> String {
            n.as_ref().map_or("{".into(), |n| { let n = n.borrow();
                let s = ((b'a' + (n.val as u8)) as char).to_string() + &s;
                if n.left.is_none() && n.right.is_none() { s }
                else { dfs(&n.left, s.clone()).min(dfs(&n.right, s)) }
            })
        }
        dfs(&root, "".into())
    }

```

