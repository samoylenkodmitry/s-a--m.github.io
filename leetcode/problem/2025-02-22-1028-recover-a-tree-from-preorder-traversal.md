---
layout: leetcode-entry
title: "1028. Recover a Tree From Preorder Traversal"
permalink: "/leetcode/problem/2025-02-22-1028-recover-a-tree-from-preorder-traversal/"
leetcode_ui: true
entry_slug: "2025-02-22-1028-recover-a-tree-from-preorder-traversal"
---

[1028. Recover a Tree From Preorder Traversal](https://leetcode.com/problems/recover-a-tree-from-preorder-traversal/description/) hard
[blog post](https://leetcode.com/problems/recover-a-tree-from-preorder-traversal/solutions/6454584/kotlin-rust-by-samoylenkodmitry-2gte/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22022025-1028-recover-a-tree-from?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/i7VubC_B8ds)
![1.webp](/assets/leetcode_daily_images/9f29da81.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/903

#### Problem TLDR

Recover binary tree from depth-dashes string #hard #stack

#### Intuition

Go deeper until the current depth is bigger than the previous, otherwise pop up.

Recursion was a more mind-bending to write.

#### Approach

* Rust can't resolve a type of the Vec until `push`
* c++ raw pointers are useful, for the Kotlin we have to resort to some wrapper to maintain the main pointer

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun recoverFromPreorder(t: String, pd: Int = -1, i: IntArray = intArrayOf(0)): TreeNode? {
        var j = i[0]; var d = 0; while (j < t.length && t[j] == '-') { d++; j++ }
        if (pd >= d) return null else i[0] = j
        var v = 0; while (i[0] < t.length && t[i[0]] != '-') v = v * 10 + (t[i[0]++] - '0')
        return TreeNode(v).apply { left = recoverFromPreorder(t, d, i); right = recoverFromPreorder(t, d, i) }
    }

```
```rust

    pub fn recover_from_preorder(t: String) -> Option<Rc<RefCell<TreeNode>>> {
        let (mut i, mut q, mut f, t) = (0, vec![], vec![], t.as_bytes());
        while i < t.len() {
            let (mut d, mut v) = (0, 0); while t[i] == b'-' { d += 1; i += 1 }
            while i < t.len() && t[i] != b'-' { v = v * 10 + (t[i] - b'0') as i32; i += 1 }
            while q.last().is_some_and(|x| *x >= d) { q.pop(); f.pop(); };
            let n = Some(Rc::new(RefCell::new(TreeNode::new(v)))); f.push(n.clone()); q.push(d); let l = f.len();
            if let Some(mut p) = f.get_mut(l - 2).and_then(|x| x.as_mut()).map(|x| x.borrow_mut()) {
                if p.left.is_none() { p.left = n.clone() } else { p.right = n.clone() }
            }
        }; f[0].clone()
    }

```
```c++

    TreeNode* recoverFromPreorder(string& t, int pd = -1, int* i = new int(0)) {
        int d = 0, v = 0, j = *i; while (j < size(t) && t[j] == '-') d++, j++;
        if (pd >= d) return nullptr; *i = j;
        while (*i < size(t) && t[*i] != '-') v = v * 10 + t[(*i)++] - '0';
        return new TreeNode(v, recoverFromPreorder(t, d, i), recoverFromPreorder(t, d, i));
    }

```

